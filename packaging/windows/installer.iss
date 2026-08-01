; Inno Setup script for whisper-flow.
;
; Produces one setup .exe. The app itself stays a folder build rather than a
; single-file one on purpose: a one-file PyInstaller binary re-extracts its
; whole payload to a temp directory on every launch, and the overlay is
; launched as a child process each time recording starts - so that cost would
; be paid on every dictation, against a budget of about 150ms.

#define AppName        "WhisperFlow"
#define AppExeName     "whisper-flow.exe"
#define AppPublisher   "Sammy Guergachi"
#define AppURL         "https://github.com/sguergachi/whisper-flow-linux"
#ifndef AppVersion
  #define AppVersion   "0.0.0"
#endif

[Setup]
AppId={{8B9C1F2E-7A44-4E1D-9C33-2F5A6D8E4B17}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppSupportURL={#AppURL}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
OutputBaseFilename=whisper-flow-{#AppVersion}-setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; Per-user by default: the app runs inside a desktop session and needs no
; machine-wide privileges. Asking for admin would be a lie about what it does.
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; The overlay needs composition attributes added in Windows 11 22H2.
MinVersion=10.0.22621
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\{#AppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "startup"; Description: "Start {#AppName} when I sign in"; \
    GroupDescription: "Startup"; Flags: checkedonce

[Files]
Source: "dist\whisper-flow\*"; DestDir: "{app}"; \
    Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}";              Filename: "{app}\{#AppExeName}"
Name: "{group}\Uninstall {#AppName}";    Filename: "{uninstallexe}"
Name: "{userstartup}\{#AppName}";        Filename: "{app}\{#AppExeName}"; \
    Tasks: startup

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Start {#AppName} now"; \
    Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Written at runtime, so the uninstaller has to name them explicitly.
Type: filesandordirs; Name: "{localappdata}\whisper-flow\logs"

[Code]
// Settings and the API key are deliberately left behind on uninstall, and
// removed only if the user asks, so reinstalling does not lose them.
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  ConfigDir: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    ConfigDir := ExpandConstant('{localappdata}\whisper-flow');
    if DirExists(ConfigDir) then
      if MsgBox('Remove your WhisperFlow settings as well?' + #13#10 +
                'Choose No to keep them for a future reinstall.',
                mbConfirmation, MB_YESNO) = IDYES then
        DelTree(ConfigDir, True, True, True);
  end;
end;
