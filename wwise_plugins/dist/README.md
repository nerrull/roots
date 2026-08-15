# Built authoring plug-ins

Windows authoring DLLs (Release config, vc170/Wwise 2025.1.10 SDK) for all six
plug-ins, built per the instructions in `../README.md`. Each `.dll` is paired
with the `.xml` property definition copied alongside it during the build.

To install into a Wwise Authoring instance, copy both files for a plug-in into
`<Wwise install>\Authoring\x64\Release\bin\Plugins\` (or the matching `Debug`
folder) and restart Wwise Authoring.
