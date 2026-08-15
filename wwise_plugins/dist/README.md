# Built authoring plug-ins

Windows authoring DLLs (Release config, vc170/Wwise 2025.1.10 SDK) for all six
plug-ins, built per the instructions in `../README.md`. Each `.dll` is paired
with the `.xml` property definition copied alongside it during the build.

To install into a Wwise Authoring instance, copy both files for a plug-in into
`<Wwise install>\Authoring\x64\Release\bin\Plugins\` (or the matching `Debug`
folder) and restart Wwise Authoring.

## Factory presets

`FactoryAssets/<Plugin>/` has 4-5 starter presets per plug-in. To install,
copy a plug-in's `FactoryAssets/<Plugin>` folder (the whole thing, including
`Manifest.xml`) into
`<Wwise install>\Authoring\Data\Factory Assets\<Plugin>\`, then in Wwise
Authoring: **Project > Import Factory Assets from...** and point it at your
Wwise install's `Factory Assets` folder (or the plug-in's `Manifest.xml`
directly).

For the four effect plug-ins (Racine Comb, Modal Resonator, Granular Texture,
Modal Voice) the presets are ShareSet-style Effect presets, in the same format
Audiokinetic ships for their own third-party plug-in bundles (verified against
McDSP's factory presets) -- these should import cleanly.

For the two source plug-ins (Macro Oscillator, Drum Synth) the presets are
full Sound objects (source plug-ins can't be presets on their own; they need
a Sound to live in), referencing "Master Audio Bus" and "Default Conversion
Settings" by the same fixed IDs Audiokinetic's own SynthOne factory presets
use. This format is **not verified against a live Wwise import** -- if it
doesn't import cleanly, the parameter values documented in each preset's XML
are still valid starting points to punch in by hand on a Sound's Source
Editor.
