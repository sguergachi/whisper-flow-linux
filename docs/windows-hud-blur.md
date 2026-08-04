# Blur on Windows

The overlay's glass comes free on Wayland: the compositor blurs what is
behind the surface and the pill tints it. Windows gives nothing away, and
finding the arrangements that work took an entire session of dead ends.

What ships today, both through `SetWindowCompositionAttribute`:

  * **The settings window** has real acrylic. It is a rectangle, which is the
    shape Windows will put a material behind, so it gets the full effect.
  * **The pill** has acrylic in a rounded rectangle rather than a capsule,
    and is smaller on Windows than on Wayland to make that fixed radius read
    as round. A window region does not clip a material - that is the whole
    constraint, and everything below follows from it.

A true capsule with blur needs WinUI 3, which is the rest of this document:
proven in a throwaway spike, not built.

## The setting that wasted the day

**Battery saver forces every acrylic and Mica surface to a flat fallback
colour, system-wide.** Not the material, not the API - a power setting. It
silently explains every flat result: DWM's acrylic, the accent policy,
`CreateHostBackdropBrush` returning black, and WinUI's own backdrops.

Check it properly. `PowerLineStatus` is the wrong flag - battery saver can
be on while plugged in:

```csharp
GetSystemPowerStatus(out var status);
bool batterySaver = status.SystemStatusFlag == 1;   // this one
```

Transparency effects (Settings > Accessibility > Visual effects) do the same
thing and are worth checking at the same time.

## What works

Four pieces, all of them necessary.

**1. The capsule.** `Microsoft.UI.Xaml.Controls.SystemBackdropElement` is the
only thing on Windows that puts a system material inside a shape. Give it a
`CornerRadius` of half its height and it is a capsule. Every other backdrop
Windows offers is the window's rectangle and nothing else.

**2. The material, via a controller.** The built-in `DesktopAcrylicBackdrop`
takes no parameters, and in Windows App SDK 2.3.1 it has no `Kind` either.
The controller has both. Drive it from a `SystemBackdrop` subclass, which is
handed the element's target - this is what the Files app does, and copying it
is what finally produced a blur:

```csharp
protected override void OnTargetConnected(
    ICompositionSupportsSystemBackdrop target, XamlRoot xamlRoot)
{
    base.OnTargetConnected(target, xamlRoot);
    var configuration = GetDefaultSystemBackdropConfiguration(target, xamlRoot);
    configuration.IsInputActive = true;
    var controller = new DesktopAcrylicController { Kind = DesktopAcrylicKind.Thin };
    controller.SetSystemBackdropConfiguration(configuration);   // before
    controller.AddSystemBackdropTarget(target);                 // after
}
```

Three details, each of which alone produces a flat fallback colour instead of
a material:

- `GetDefaultSystemBackdropConfiguration`, not a hand-built
  `SystemBackdropConfiguration`. The base class wires the real one to the
  target's theme and activation.
- `SetSystemBackdropConfiguration` **before** `AddSystemBackdropTarget`. A
  target attached to a controller with no configuration has nothing to draw.
- `IsInputActive = true`. The default follows window activation, and an
  overlay never takes focus - without this it sits at its fallback colour for
  its entire life.

`Kind = Thin` is the least frosted of the acrylics.

**3. A transparent window.** WinUI 3 has no transparent backdrop of its own -
there is no `TransparentBackdrop` type in 2.3.1 - and no DWM trick fixes it,
because WinUI renders into its own swapchain and the alpha never reaches DWM.
WinUIEx's `TransparentTintBackdrop` does it.

**4. No frame, no shadow.** `OverlappedPresenter.SetBorderAndTitleBar(false,
false)` removes only what WinUI draws. The hairline and the shadow are the
non-client area, which is DWM's:

```csharp
// strip to a bare popup
style &= ~(WS_CAPTION | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX
           | WS_SYSMENU | WS_BORDER | WS_DLGFRAME);
style |= WS_POPUP;
SetWindowLongPtr(hwnd, GWL_STYLE, style);
DwmSetWindowAttribute(hwnd, DWMWA_NCRENDERING_POLICY, DWMNCRP_DISABLED);
SetWindowPos(..., SWP_FRAMECHANGED);   // or the style change is not recalculated
```

Also: `DWMWA_BORDER_COLOR = DWMWA_COLOR_NONE` and
`DWMWA_WINDOW_CORNER_PREFERENCE = DWMWCP_DONOTROUND`.

## Two traps

**`AppWindow.Resize` and `Move` take physical pixels; XAML is in logical
units.** Sizing the window from the same numbers as the content makes it half
the size it needs on a 2x display, and the capsule is clipped into what looks
like a plain rectangle.

**Build framework-dependent**, not self-contained. `WindowsAppSDKSelfContained`
was tried both ways; the material needs the framework package.

## What does not work, and why

| Approach | Result |
| --- | --- |
| `DWMWA_SYSTEMBACKDROP_TYPE` | Draws into the window *frame*, so a toolkit that paints its own client area never shows it: applied cleanly, reported success, looked like nothing. Needs `DwmExtendFrameIntoClientArea` with margins of -1 first. Even then it ignores `SetWindowRgn` - the region was applied, `GetWindowRgnBox` confirmed the capsule, and the slab stayed. |
| `SetWindowCompositionAttribute` accent policy | **This is what works.** Blurs the client area with a tint we control. Undocumented, and the only call that does this to a plain Win32 window. Also ignores the window region, which is why the pill is a rounded rectangle and not a capsule. |
| `DwmEnableBlurBehindWindow` | Has not blurred anything since Windows 8 removed Aero. Still useful for one thing: over an *empty* region it makes DWM honour the window's alpha channel, which is how the GTK overlay gets transparent corners today. |
| `CreateHostBackdropBrush` | Renders black on a plain Win32 window. Effectively UWP-only. |
| `CreateBackdropBrush` | Samples only its own composition tree, which for a window holding nothing else is nothing. |
| `DesktopAcrylicController` attached straight to a `SystemBackdropElement` | The cast fails; the element does not implement `ICompositionSupportsSystemBackdrop`. Go through a `SystemBackdrop` subclass instead. |
| Mica, at any opacity | Tints from the wallpaper and does not blur what is behind the window. Calm, and not the effect. |
| Capturing and blurring the screen ourselves | Works, and is a photograph. Live needs `WDA_EXCLUDEFROMCAPTURE` so the capture does not contain the pill, which also hides the overlay from screen recordings. Removed in `45ddcf8`; recoverable from its parent. |

## The cost of adopting it

The overlay would be a WinUI 3 app on Windows and GTK on Linux - two
implementations of the pill. The Windows App SDK runtime ships in the
installer. The squircle, waveform, sheen and inner shadow are cairo drawing
today and would be redone in XAML or Win2D, and `CornerRadius` gives a true
capsule with circular caps rather than the continuous-curvature squircle
(`SQUIRCLE_N = 2.3`) the pill is drawn with now.

The process model survives unchanged: the overlay is already a separate
process the daemon drives over stdin.
