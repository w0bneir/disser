# Удалённое выполнение на RTX 5070

## Роли компьютеров

- `D:\Projects\disser` на основном ПК — единственное место редактирования кода.
- `C:\Users\godmi\Projects\disser` на RTX 5070 (`pspc`) — чистая исполнительная
  копия: только `git pull --ff-only`, тесты и запуски.
- `D:\YandexDisk\SFX_artifacts` — архив завершённых результатов. Рабочая папка,
  Conda-среда и Hugging Face cache не должны находиться в Yandex Disk.

## Локальная рабочая копия

Архивная папка `D:\YandexDisk\disser` не используется для разработки. Если новая
копия ещё не создана, на основном ПК выполните:

```bat
mkdir D:\Projects
cd /d D:\Projects
git clone https://github.com/w0bneir/disser.git disser
code D:\Projects\disser
```

До начала изменения кода используйте VS Code Source Control → **Pull**. После
логически законченного изменения: **Commit** → **Sync Changes**. На RTX 5070
изменения кода не вносятся — перед запуском там разрешён только
`git pull --ff-only`.

## Канал удалённой работы: Tailscale + SSH

Эти действия выполняются один раз. На обоих ПК установите Tailscale, войдите в один
и тот же tailnet и убедитесь, что RTX 5070 имеет MagicDNS-имя (или Tailscale IP).

На RTX 5070 используется существующий пользователь `godmi`. Устанавливать OpenSSH
Server нужно из PowerShell **от имени администратора**:

```powershell
Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH*'
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Start-Service sshd
Set-Service -Name sshd -StartupType Automatic
Get-NetFirewallRule -Name OpenSSH-Server-In-TCP
```

На основном ПК один раз создайте отдельный ключ. Команда интерактивно спросит
passphrase; её лучше задать и сохранить в менеджере ключей Windows:

```powershell
ssh-keygen -t ed25519 -a 100 -f "$env:USERPROFILE\.ssh\id_ed25519_disser_5070" -C "disser-control"
Get-Content "$env:USERPROFILE\.ssh\id_ed25519_disser_5070.pub"
```

Скопируйте одну выведенную публичную строку в
`C:\Users\godmi\.ssh\authorized_keys` на RTX 5070. На RTX 5070, войдя под
`godmi`, выполните:

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.ssh"
notepad "$env:USERPROFILE\.ssh\authorized_keys"
icacls "$env:USERPROFILE\.ssh" /inheritance:r /grant:r "${env:USERNAME}:(OI)(CI)F" /grant:r "SYSTEM:(OI)(CI)F"
icacls "$env:USERPROFILE\.ssh\authorized_keys" /inheritance:r /grant:r "${env:USERNAME}:F" /grant:r "SYSTEM:F"
```

На основном ПК добавьте в `C:\Users\godmi\.ssh\config`:

```text
Host disser-5070
    HostName pspc
    User godmi
    IdentityFile C:\Users\godmi\.ssh\id_ed25519_disser_5070
    IdentitiesOnly yes
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

Проверьте `ssh disser-5070`. Только после успешного входа ключом ограничьте
стандартное правило SSH Tailscale-сетью на RTX 5070:

```powershell
Set-NetFirewallRule -Name OpenSSH-Server-In-TCP -RemoteAddress 100.64.0.0/10
```

В VS Code на основном ПК установите расширение **Remote - SSH**, выберите
`Remote-SSH: Connect to Host...` → `disser-5070` и откройте
`C:\Users\godmi\Projects\disser`. В открытом удалённом окне не редактируйте
код: используйте его только для терминалов, диагностики и запуска.

## Создание среды на RTX 5070

В Miniconda Prompt под пользователем `godmi`:

```bat
mkdir C:\Users\godmi\Projects
cd /d C:\Users\godmi\Projects
git clone https://github.com/w0bneir/disser.git
cd disser
conda create -n sfx_gen_5070 python=3.11 -y
conda activate sfx_gen_5070
python -m pip install --upgrade pip
python -m pip install -r requirements\windows-cu128-blackwell.txt
python -m pip install -r requirements\base.txt
```

Не используйте корневой `requirements.txt` на RTX 5070: он предназначен для
старой CUDA 12.1 среды GTX 1070.

Чтобы вынести кэш модели на локальный SSD с запасом 30 ГБ:

```bat
setx HF_HOME D:\AI\hf-cache
```

Закройте и снова откройте терминал после `setx`.

## Предзапусковая последовательность

Выполнять на RTX 5070 перед первым скачиванием модели:

```bat
conda activate sfx_gen_5070
cd /d C:\Users\godmi\Projects\disser
git pull --ff-only
powershell -ExecutionPolicy Bypass -File tools\preflight_gpu.ps1
python run_stable_audio_experiments.py --preflight-only
python -m pip check
python verify_setup.py
python -B -m unittest -v test_stable_audio_guidance.py
```

Только затем допускается один baseline с `--allow-download`. Ручной guidance
запускается лишь после успешного baseline и preflight с 12 000 MiB total /
10 000 MiB free VRAM.

## Эксперимент и артефакты

Для каждого запуска создавайте отдельную папку, например:

```bat
set RUN_ID=2026-08-16_rtx5070_smoke_01
mkdir results\%RUN_ID%
```

В первом удалённом терминале запустите мониторинг:

```bat
powershell -ExecutionPolicy Bypass -File tools\monitor_gpu.ps1 -OutputPath results\%RUN_ID%\gpu_monitor.csv
```

Во втором терминале запускайте ровно один эксперимент. После успешного
завершения сохраните контекст:

```bat
powershell -ExecutionPolicy Bypass -File tools\capture_run_context.ps1 -ResultsDirectory results\%RUN_ID%
```

С основного ПК заберите готовую папку:

```powershell
scp -r disser-5070:"C:/Users/godmi/Projects/disser/results/<RUN_ID>" "D:\YandexDisk\SFX_artifacts\"
```

Не копируйте в Git файлы из `results/`.

## Безопасность

- На GTX 1070 и RTX 3070 Laptop не запускать ручной Stable Audio guidance.
- До каждого запуска проверять `tools/preflight_gpu.ps1`.
- Запускать только одну пару `baseline/guided`; не использовать GPU одновременно
  для игры, рендеринга или иных тяжёлых задач.
- Сначала пройти последовательность: import → baseline → 4 шага → 10 → 20 → 50.
