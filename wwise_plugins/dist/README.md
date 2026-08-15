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

**Schema note:** an earlier version of these files used `SchemaVersion="97"`
and (for the two source plug-ins) the wrong root element and a missing set of
`<Reference>` attributes, copied from a stale ~2019.2-era Audiokinetic example.
That caused a crash on import. The current files are checked against this
machine's actual Wwise 2025.1.10 install (`Authoring/Data/Factory
Assets/SynthOne` and `.../Effects/Factory Effects.wwu`, i.e. Audiokinetic's own
shipped files for *this* Wwise version, not an old one) --
`SchemaVersion="133"`, `<Effects>` root for effect presets, `<Containers>` root
for source-plugin Sound presets, and the `CompanyID="4095" PluginID="65535"
PluginType="15"` attributes on `<Reference>` elements. This has **not** been
confirmed by an actual successful import in a running Wwise Authoring instance
-- only by matching the current install's own file structure byte-for-byte
where comparable. Before importing into a real project: **commit/back up your
Wwise project first.**

For the four effect plug-ins (Racine Comb, Modal Resonator, Granular Texture,
Modal Voice) the presets are ShareSet-style Effect presets -- no external
object references, lower risk.

For the two source plug-ins (Macro Oscillator, Drum Synth) the presets are
full Sound objects (source plug-ins can't be presets on their own; they need a
Sound to live in), referencing "Main Audio Bus" and "Default Conversion
Settings" by the fixed IDs Audiokinetic's own SynthOne factory presets use for
this Wwise version. If it still doesn't import cleanly, the parameter values
documented in each preset's XML are valid starting points to punch in by hand
on a Sound's Source Editor instead.
