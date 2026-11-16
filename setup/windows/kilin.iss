; KíLín installer for 64-bit Windows. Build it with:
;   powershell -NoProfile -ExecutionPolicy Bypass -File setup\windows\build_installer.ps1 [-Wheel]
;
; The installer holds only uv, an icon and, with -Wheel, a wheel of this
; checkout. While installing, uv downloads Python 3.12 and installs KíLín
; (kilin[app]) into a versioned runtime under %LOCALAPPDATA%\kilin\runtime,
; choosing the PyTorch build for the computer's GPU. Models are downloaded
; later, inside the application. No administrator rights are needed.

#ifndef AppVersion
  #error Build with setup\windows\build_installer.ps1, which defines AppVersion.
#endif

[Setup]
; Keep this ID: Windows uses it to recognise updates of the same app.
AppId={{A4555CEA-915D-4237-998D-FDB55F170A4E}
AppName=KíLín
AppVersion={#AppVersion}
AppPublisher=KíLín developers
DefaultDirName={localappdata}\Programs\KiLin
DisableDirPage=yes
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=KiLin-Setup-{#AppVersion}
SetupIconFile=kilin.ico
UninstallDisplayIcon={app}\kilin.ico
UninstallDisplayName=KíLín
LicenseFile=..\..\LICENSE
WizardStyle=modern
Compression=lzma2
SolidCompression=yes
; The repair-command variables below change the user environment.
ChangesEnvironment=yes
; RedirectionGuard, inherited by every program Setup starts, blocks the
; folder link uv makes for Python 3.12 (error 448, "untrusted mount
; point"). It guards elevated installers against link tricks in shared
; folders; this one runs as the user and writes only to the user's folders.
RedirectionGuard=no

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Files]
Source: "{#StageDir}\uv.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "kilin.ico"; DestDir: "{app}"; Flags: ignoreversion
#ifdef WheelFile
Source: "{#StageDir}\{#WheelFile}"; DestDir: "{app}"; Flags: ignoreversion
#endif

[Icons]
Name: "{autoprograms}\KíLín"; Filename: "{code:RuntimeDir}\Scripts\kilin-gui.exe"; IconFilename: "{app}\kilin.ico"
Name: "{autodesktop}\KíLín"; Filename: "{code:RuntimeDir}\Scripts\kilin-gui.exe"; IconFilename: "{app}\kilin.ico"; Tasks: desktopicon

[Registry]
; The application offers backend-repair commands that must use this uv and
; this exact KíLín build, as the macOS launcher does.
Root: HKCU; Subkey: "Environment"; ValueType: string; ValueName: "KILIN_APP_UV"; ValueData: "{app}\uv.exe"; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Environment"; ValueType: string; ValueName: "KILIN_APP_SPEC"; ValueData: "{code:Spec}"; Flags: uninsdeletevalue

[Run]
Filename: "{code:RuntimeDir}\Scripts\kilin-gui.exe"; Description: "Open KíLín"; Flags: postinstall nowait skipifsilent

[UninstallDelete]
; Only what the installer made; projects and preferences are kept.
Type: filesandordirs; Name: "{localappdata}\kilin\runtime"

[Code]
var
  { Shows the install step and uv's latest output line while uv runs. }
  ProgressPage: TOutputMarqueeProgressPage;
  InstallLog, StepText: String;

procedure InitializeWizard();
begin
  ProgressPage := CreateOutputMarqueeProgressPage('Installing Python and KíLín',
    'This takes several minutes and needs an internet connection.');
end;

function RuntimeRoot(): String;
begin
  Result := ExpandConstant('{localappdata}\kilin\runtime');
end;

{ One folder per version, so an update never runs half-installed code. }
function RuntimeDir(Param: String): String;
begin
  Result := RuntimeRoot() + '\{#AppVersion}';
end;

{ What is installed, from PyPI or from the bundled wheel. }
function Spec(Param: String): String;
begin
#ifdef WheelFile
  Result := ExpandConstant('{app}\{#WheelFile}') + '[app]';
#else
  Result := 'kilin[app]=={#AppVersion}';
#endif
end;

{ Each line uv prints: keep it in the log and show it under the step. }
procedure ShowUvLine(const S: String; const Error, FirstLine: Boolean);
begin
  SaveStringToFile(InstallLog, S + #13#10, True);
  ProgressPage.SetText(StepText, S);
  ProgressPage.Animate();
end;

{ Run uv hidden for one step; True when it succeeds. }
function RunUv(const Uv, Arguments, Step: String): Boolean;
var
  Code: Integer;
  Command: String;
begin
  StepText := Step;
  ProgressPage.SetText(Step, '');
  Command := 'set "UV_PYTHON_INSTALL_DIR=' + RuntimeRoot() + '\python"' +
    ' && set "UV_CACHE_DIR=' + RuntimeRoot() + '\uv-cache"' +
    ' && "' + Uv + '" ' + Arguments;
  Result := ExecAndLogOutput(ExpandConstant('{cmd}'), '/C "' + Command + '"',
    '', SW_HIDE, ewWaitUntilTerminated, Code, @ShowUvLine) and (Code = 0);
end;

{ Install Python and KíLín before any file is copied. A failure stops Setup
  with a message and leaves an earlier installation untouched. }
function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  Uv, Runtime, Source: String;
  Installed: Boolean;
begin
  Result := '';
  ExtractTemporaryFile('uv.exe');
  Uv := ExpandConstant('{tmp}\uv.exe');
#ifdef WheelFile
  ExtractTemporaryFile('{#WheelFile}');
  Source := ExpandConstant('{tmp}\{#WheelFile}') + '[app]';
#else
  Source := Spec('');
#endif
  Runtime := RuntimeDir('');
  InstallLog := RuntimeRoot() + '\install.log';
  ForceDirectories(RuntimeRoot());
  DelTree(Runtime, True, True, True);
  DeleteFile(InstallLog);
  ProgressPage.Show();
  try
    { --torch-backend auto picks the CUDA build of PyTorch for an NVIDIA
      GPU and the CPU build otherwise; PyPI alone has only the CPU build. }
    Installed :=
      RunUv(Uv, 'venv --python 3.12 "' + Runtime + '"',
        'Step 1 of 2: Getting Python 3.12') and
      RunUv(Uv, 'pip install --torch-backend auto --python "' + Runtime +
          '\Scripts\python.exe" "' + Source + '"',
        'Step 2 of 2: Installing KíLín and its packages') and
      FileExists(Runtime + '\Scripts\kilin-gui.exe');
  finally
    ProgressPage.Hide();
  end;
  if not Installed then
  begin
    DelTree(Runtime, True, True, True);
    Result := 'KíLín could not be installed. Check the internet ' +
      'connection and run the installer again. Details: ' + InstallLog;
  end;
  DelTree(RuntimeRoot() + '\uv-cache', True, True, True);
end;

{ After a successful update, remove the runtimes of other versions. A
  runtime still in use stays and is removed by a later update. }
procedure CurStepChanged(CurStep: TSetupStep);
var
  Found: TFindRec;
begin
  if CurStep <> ssPostInstall then
    Exit;
  if FindFirst(RuntimeRoot() + '\*', Found) then
  try
    repeat
      if (Found.Attributes and FILE_ATTRIBUTE_DIRECTORY <> 0)
          and (Found.Name <> '.') and (Found.Name <> '..')
          and (Found.Name <> 'python') and (Found.Name <> '{#AppVersion}') then
        DelTree(RuntimeRoot() + '\' + Found.Name, True, True, True);
    until not FindNext(Found);
  finally
    FindClose(Found);
  end;
end;
