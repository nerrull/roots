--[[----------------------------------------------------------------------------
The content of this file includes portions of the AUDIOKINETIC Wwise Technology
released in source code form as part of the SDK installer package.

Commercial License Usage

Licensees holding valid commercial licenses to the AUDIOKINETIC Wwise Technology
may use this file in accordance with the end user license agreement provided
with the software or, alternatively, in accordance with the terms contained in a
written agreement between you and Audiokinetic Inc.

Apache License Usage

Alternatively, this file may be used under the Apache License, Version 2.0 (the
"Apache License"); you may not use this file except in compliance with the
Apache License. You may obtain a copy of the Apache License at
http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed
under the Apache License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
OR CONDITIONS OF ANY KIND, either express or implied. See the Apache License for
the specific language governing permissions and limitations under the License.

  Copyright (c) 2026 Audiokinetic Inc.
------------------------------------------------------------------------------]]

if not _AK_PREMAKE then
    error('You must use the custom Premake5 scripts by adding the following parameter: --scripts="Scripts\\Premake"', 1)
end

local MI_DIR = os.getenv("MI_EURORACK_DIR")
    or path.getabsolute(_OPTIONS["plugindir"] .. "/../../../eurorack")

-- The plug-in only exposes the four percussion models, but processors.cc holds
-- a static dispatch table naming every Peaks function, so the control-rate
-- processors have to be linked in as well or the table is left unresolved.
local MI_FILES =
{
    MI_DIR .. "/peaks/processors.cc",
    MI_DIR .. "/peaks/drums/bass_drum.cc",
    MI_DIR .. "/peaks/drums/snare_drum.cc",
    MI_DIR .. "/peaks/drums/high_hat.cc",
    MI_DIR .. "/peaks/drums/fm_drum.cc",
    MI_DIR .. "/peaks/modulations/lfo.cc",
    MI_DIR .. "/peaks/modulations/multistage_envelope.cc",
    MI_DIR .. "/peaks/number_station/number_station.cc",
    MI_DIR .. "/peaks/pulse_processor/pulse_randomizer.cc",
    MI_DIR .. "/peaks/pulse_processor/pulse_shaper.cc",
    MI_DIR .. "/peaks/resources.cc",
    MI_DIR .. "/stmlib/utils/random.cc",
}

-- TEST selects stmlib's portable code paths.
local MI_DEFINES = { "TEST" }

local Plugin = {}
Plugin.name = "DrumSynth"
Plugin.factoryheader = "../SoundEnginePlugin/DrumSynthSourceFactory.h"
Plugin.appleteamid = ""
Plugin.signtoolargs = {}
Plugin.sdk = {}
Plugin.sdk.static = {}
Plugin.sdk.shared = {}
Plugin.authoring = {}

-- SDK STATIC PLUGIN SECTION
Plugin.sdk.static.includedirs = -- https://github.com/premake/premake-core/wiki/includedirs
{
    MI_DIR,
}
Plugin.sdk.static.files = -- https://github.com/premake/premake-core/wiki/files
{
    "**.cpp",
    "**.h",
    "**.hpp",
    "**.c",
    table.unpack(MI_FILES),
}
Plugin.sdk.static.excludes = -- https://github.com/premake/premake-core/wiki/removefiles
{
    "DrumSynthSourceShared.cpp"
}
Plugin.sdk.static.links = -- https://github.com/premake/premake-core/wiki/links
{
}
Plugin.sdk.static.libsuffix = "Source"
Plugin.sdk.static.libdirs = -- https://github.com/premake/premake-core/wiki/libdirs
{
}
Plugin.sdk.static.defines = -- https://github.com/premake/premake-core/wiki/defines
{
    table.unpack(MI_DEFINES),
}

-- SDK SHARED PLUGIN SECTION
Plugin.sdk.shared.includedirs =
{
}
Plugin.sdk.shared.files =
{
    "DrumSynthSourceShared.cpp",
    "DrumSynthSourceFactory.h",
}
Plugin.sdk.shared.excludes =
{
}
Plugin.sdk.shared.links =
{
}
Plugin.sdk.shared.libdirs =
{
}
Plugin.sdk.shared.defines =
{
}

-- AUTHORING PLUGIN SECTION
Plugin.authoring.includedirs =
{
    MI_DIR,
}
Plugin.authoring.files =
{
    "**.cpp",
    "**.h",
    "**.hpp",
    "**.c",
    "DrumSynth.def",
    "DrumSynth.xml",
    "**.rc",
}
Plugin.authoring.excludes =
{
}
Plugin.authoring.links =
{
}
Plugin.authoring.libdirs =
{
}
Plugin.authoring.defines =
{
    table.unpack(MI_DEFINES),
}
--[[DISABLED SECTION BEGIN
-- SDK TEST PLUGIN SECTION
_CATCH2_DIR = _AK_SDK_ROOT .. "samples/3rdParty/Catch2/extras/"

Plugin.sdk.test = {}

Plugin.sdk.test.includedirs =
{
    _CATCH2_DIR
}
Plugin.sdk.test.files =
{
    "**.cpp",
    "**.h",
    "**.hpp",
    "**.c",
    _CATCH2_DIR .. "catch_amalgamated.cpp"
}
Plugin.sdk.test.excludes =
{
}
Plugin.sdk.test.custom = function()

    filter { "system:android" }
        links {
            "log"
        }
end
Plugin.sdk.test.links =
{
}
Plugin.sdk.test.libdirs =
{
}
Plugin.sdk.test.defines =
{
    "NOMINMAX"
}
DISABLED SECTION END--]]
return Plugin
