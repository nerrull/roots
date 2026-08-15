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

-- Elements' exciter, resonator and reverb.
local MI_FILES =
{
    MI_DIR .. "/elements/dsp/part.cc",
    MI_DIR .. "/elements/dsp/exciter.cc",
    MI_DIR .. "/elements/dsp/multistage_envelope.cc",
    MI_DIR .. "/elements/dsp/ominous_voice.cc",
    MI_DIR .. "/elements/dsp/resonator.cc",
    MI_DIR .. "/elements/dsp/string.cc",
    MI_DIR .. "/elements/dsp/tube.cc",
    MI_DIR .. "/elements/dsp/voice.cc",
    MI_DIR .. "/elements/resources.cc",
    MI_DIR .. "/stmlib/dsp/atan.cc",
    MI_DIR .. "/stmlib/dsp/units.cc",
    MI_DIR .. "/stmlib/utils/random.cc",
}

-- TEST selects stmlib's portable code paths.
-- NOMINMAX and _USE_MATH_DEFINES: the MI DSP code was only ever compiled with
-- Clang/GCC before. Without NOMINMAX, windows.h's min/max macros corrupt every
-- std::min/std::max call in the MI sources; without _USE_MATH_DEFINES, MSVC's
-- <cmath> does not define M_PI.
local MI_DEFINES = { "TEST", "NOMINMAX", "_USE_MATH_DEFINES" }

local Plugin = {}
Plugin.name = "ModalVoice"
Plugin.factoryheader = "../SoundEnginePlugin/ModalVoiceFXFactory.h"
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
    "ModalVoiceFXShared.cpp"
}
Plugin.sdk.static.links = -- https://github.com/premake/premake-core/wiki/links
{
}
Plugin.sdk.static.libsuffix = "FX"
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
    "ModalVoiceFXShared.cpp",
    "ModalVoiceFXFactory.h",
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
    "ModalVoice.def",
    "ModalVoice.xml",
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
