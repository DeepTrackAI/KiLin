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
; No licence page: as on macOS, KíLín shows its terms of use on first
; launch and records the acceptance (kilin/gui/terms.txt).
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
Filename: "{localappdata}\kilin\runtime\install.log"; Description: "Show the install log (what was installed)"; Flags: postinstall shellexec nowait skipifsilent unchecked

[UninstallDelete]
; Only what the installer made; projects and preferences are kept.
Type: filesandordirs; Name: "{localappdata}\kilin\runtime"

[Code]
const
  { Where Windows keeps this app's uninstall entry, named after AppId. }
  UninstallKey = 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{A4555CEA-915D-4237-998D-FDB55F170A4E}_is1';

var
  { Shows what uv downloads, the percentage and the time left. }
  ProgressPage: TOutputProgressWizardPage;
  { One bullet per download under the bar: downloaded or downloading. }
  DownloadList: TNewMemo;
  InstallLog, Activity: String;
  { Progress in tenths of a percent: this step's range and what is shown. }
  StepLow, StepHigh, Shown: Integer;
  { The downloads uv announced in this step, with sizes in KiB. }
  Names: array of String;
  Sizes: array of Int64;
  Done: array of Boolean;
  TotalKiB, DoneKiB: Int64;
  Installing: Boolean;
  PageTick, DownloadTick, Timer: LongWord;

function GetTickCount(): LongWord;
  external 'GetTickCount@kernel32.dll stdcall';
function SetTimer(Wnd, Id, Elapse, TimerFunc: LongWord): LongWord;
  external 'SetTimer@user32.dll stdcall';
function KillTimer(Wnd, Id: LongWord): BOOL;
  external 'KillTimer@user32.dll stdcall';

{ Offer to reinstall or remove an earlier installation of KíLín. }
function InitializeSetup(): Boolean;
var
  Version, Uninstaller, Install: String;
  Code: Integer;
begin
  Result := True;
  if WizardSilent() or
      not RegQueryStringValue(HKCU, UninstallKey, 'UninstallString', Uninstaller) then
    Exit;
  if not RegQueryStringValue(HKCU, UninstallKey, 'DisplayVersion', Version) then
    Version := '';
  if Version = '{#AppVersion}' then
    Install := 'Reinstall KíLín ' + Version + #13#10 +
      'Install it again, for example to repair it.'
  else
    Install := 'Install KíLín {#AppVersion}' + #13#10 +
      'Replace the installed version.';
  case TaskDialogMsgBox('KíLín ' + Version + ' is already installed.',
      'Projects and preferences are kept either way.', mbConfirmation,
      MB_YESNOCANCEL, [Install, 'Remove KíLín' + #13#10 +
        'Uninstall the program and its Python packages.'], 0) of
    IDYES:
      Result := True;
    IDNO:
      begin
        Exec(RemoveQuotes(Uninstaller), '', '', SW_SHOW,
          ewWaitUntilTerminated, Code);
        Result := False;
      end;
  else
    Result := False;
  end;
end;

procedure InitializeWizard();
begin
  ProgressPage := CreateOutputProgressPage('Installing Python and KíLín',
    'This takes several minutes and needs an internet connection.');
  DownloadList := TNewMemo.Create(ProgressPage);
  DownloadList.Parent := ProgressPage.Surface;
  DownloadList.Top := ProgressPage.ProgressBar.Top +
    ProgressPage.ProgressBar.Height + ScaleY(12);
  DownloadList.Width := ProgressPage.SurfaceWidth;
  DownloadList.Height := ProgressPage.SurfaceHeight - DownloadList.Top;
  DownloadList.ReadOnly := True;
  DownloadList.ScrollBars := ssVertical;
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

{ A size as uv prints it, such as '2.3GiB' or '512.0KiB', in KiB. }
function SizeKiB(const S: String): Int64;
var
  Number, Scale: Int64;
  I: Integer;
  Decimal: Boolean;
  Suffix: String;
begin
  Number := 0;
  Scale := 1;
  Decimal := False;
  I := 1;
  while I <= Length(S) do
  begin
    if S[I] = '.' then
      Decimal := True
    else if (S[I] >= '0') and (S[I] <= '9') then
    begin
      Number := Number * 10 + Ord(S[I]) - Ord('0');
      if Decimal then
        Scale := Scale * 10;
    end
    else
      Break;
    I := I + 1;
  end;
  Suffix := Copy(S, I, Length(S));
  if Suffix = 'GiB' then
    Result := Number * 1048576 div Scale
  else if Suffix = 'MiB' then
    Result := Number * 1024 div Scale
  else if Suffix = 'KiB' then
    Result := Number div Scale
  else
    Result := Number div Scale div 1024;
end;

{ 1258291 KiB as '1.2 GiB'; less than a GiB in MiB. }
function ShowSize(KiB: Int64): String;
var
  Tenths: Int64;
begin
  if KiB >= 1048576 then
  begin
    Tenths := KiB * 10 div 1048576;
    Result := ' GiB';
  end
  else
  begin
    Tenths := KiB * 10 div 1024;
    Result := ' MiB';
  end;
  Result := IntToStr(Tenths div 10) + '.' + IntToStr(Tenths mod 10) + Result;
end;

function ShowMinutes(Seconds: Int64): String;
begin
  if Seconds < 60 then
    Result := 'less than a minute'
  else
    Result := IntToStr((Seconds + 30) div 60) + ' min';
end;

{ List the downloads in the order uv started them, such as
    • torch (2.3 GiB): downloaded
    • numpy (15.0 MiB): downloading }
procedure ShowDownloads();
var
  I: Integer;
  Text, State: String;
begin
  Text := '';
  for I := 0 to GetArrayLength(Names) - 1 do
  begin
    if Done[I] then
      State := 'downloaded'
    else
      State := 'downloading';
    Text := Text + '• ' + Names[I] + ' (' + ShowSize(Sizes[I]) + '): ' +
      State + #13#10;
  end;
  DownloadList.Text := Text;
end;

{ Show the percentage, the downloaded size and an estimate of the time
  left. Downloads fill nine tenths of the step; installing them the rest. }
procedure UpdateProgress();
var
  Position: Integer;
  Elapsed: Int64;
  Heading, Details: String;
begin
  Position := StepLow;
  if Installing then
    Position := StepLow + (StepHigh - StepLow) * 9 div 10
  else if TotalKiB > 0 then
    Position := StepLow + (StepHigh - StepLow) * 9 * DoneKiB div TotalKiB div 10;
  { New downloads add to the total; the bar never moves back. }
  if Position > Shown then
    Shown := Position;
  ProgressPage.SetProgress(Shown, 1000);
  Details := IntToStr(Shown div 10) + '%';
  Heading := Activity;
  { uv names only large downloads, so the list is not every package. }
  if not Installing and (StepLow > 0) and (GetArrayLength(Names) > 0) then
    Heading := 'Downloading packages';
  if (TotalKiB > 0) and not Installing then
  begin
    Details := Details + ', ' + ShowSize(DoneKiB) + ' of ' +
      ShowSize(TotalKiB) + ' downloaded';
    Elapsed := GetTickCount() - DownloadTick;
    if (DoneKiB > 0) and (Elapsed >= 5000) then
      Details := Details + ', about ' + ShowMinutes(Elapsed *
        (TotalKiB - DoneKiB) div DoneKiB div 1000) + ' left';
  end
  else
    Details := Details + ', ' + ShowMinutes((GetTickCount() - PageTick) div 1000)
      + ' so far';
  ProgressPage.SetText(Heading, Details);
end;

{ Refresh the time every second, also while uv prints nothing. }
procedure OnTimer(Arg1, Arg2, Arg3, Arg4: LongWord);
begin
  UpdateProgress();
end;

{ The position of the last ' (' in S, or 0. }
function LastParen(const S: String): Integer;
begin
  Result := Length(S) - 1;
  while (Result > 0) and (Copy(S, Result, 2) <> ' (') do
    Result := Result - 1;
end;

{ Each line uv prints: keep it in the log and follow the downloads. uv
  prints 'Downloading torch (2.3GiB)' when a large download starts,
  'Downloaded torch' when it ends and 'Prepared 152 packages' before
  installing them; other lines are shown as they are. }
procedure ShowUvLine(const S: String; const Error, FirstLine: Boolean);
var
  Line, Name: String;
  I, Cut: Integer;
begin
  SaveStringToFile(InstallLog, S + #13#10, True);
  Line := Trim(S);
  Cut := LastParen(Line);
  if (Pos('Downloading ', Line) = 1) and (Cut > 12) then
  begin
    I := GetArrayLength(Names);
    SetArrayLength(Names, I + 1);
    SetArrayLength(Sizes, I + 1);
    SetArrayLength(Done, I + 1);
    Names[I] := Copy(Line, 13, Cut - 13);
    Sizes[I] := SizeKiB(Copy(Line, Cut + 2, Length(Line) - Cut - 2));
    Done[I] := False;
    TotalKiB := TotalKiB + Sizes[I];
    if I = 0 then
      DownloadTick := GetTickCount();
  end
  else if Pos('Downloaded ', Line) = 1 then
  begin
    Name := Copy(Line, 12, Length(Line));
    for I := 0 to GetArrayLength(Names) - 1 do
      if not Done[I] and (Pos(Name, Names[I]) = 1) then
      begin
        Done[I] := True;
        DoneKiB := DoneKiB + Sizes[I];
        Break;
      end;
  end
  else if Pos('Prepared ', Line) = 1 then
  begin
    Installing := True;
    Activity := 'Installing the downloaded packages';
  end
  else if Line <> '' then
    Activity := Line;
  ShowDownloads();
  UpdateProgress();
end;

{ Run uv hidden for one step, filling the bar from Low to High tenths of a
  percent; True when it succeeds. }
function RunUv(const Uv, Arguments, Title: String; Low, High: Integer): Boolean;
var
  Code: Integer;
  Command: String;
begin
  Activity := Title;
  StepLow := Low;
  StepHigh := High;
  SetArrayLength(Names, 0);
  SetArrayLength(Sizes, 0);
  SetArrayLength(Done, 0);
  TotalKiB := 0;
  DoneKiB := 0;
  Installing := False;
  ShowDownloads();
  UpdateProgress();
  Command := 'set "UV_PYTHON_INSTALL_DIR=' + RuntimeRoot() + '\python"' +
    ' && set "UV_CACHE_DIR=' + RuntimeRoot() + '\uv-cache"' +
    ' && "' + Uv + '" ' + Arguments;
  Result := ExecAndLogOutput(ExpandConstant('{cmd}'), '/C "' + Command + '"',
    '', SW_HIDE, ewWaitUntilTerminated, Code, @ShowUvLine) and (Code = 0);
  if Result then
    Shown := High;
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
  SaveStringToFile(InstallLog, 'KíLín {#AppVersion} from ' + Source +
    #13#10 + 'Into ' + Runtime + #13#10 + 'Lines starting with + are the ' +
    'installed packages and their versions.' + #13#10#13#10, False);
  Shown := 0;
  PageTick := GetTickCount();
  ProgressPage.Show();
  Timer := SetTimer(0, 0, 1000, CreateCallback(@OnTimer));
  try
    { --torch-backend auto picks the CUDA build of PyTorch for an NVIDIA
      GPU and the CPU build otherwise; PyPI alone has only the CPU build. }
    Installed :=
      RunUv(Uv, 'venv --python 3.12 "' + Runtime + '"',
        'Getting Python 3.12', 0, 50) and
      RunUv(Uv, 'pip install --torch-backend auto --python "' + Runtime +
          '\Scripts\python.exe" "' + Source + '"',
        'Finding the packages KíLín needs', 50, 1000) and
      FileExists(Runtime + '\Scripts\kilin-gui.exe');
  finally
    KillTimer(0, Timer);
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
