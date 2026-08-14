/*******************************************************************************
Drum Synth -- a Wwise source plug-in wrapping the Mutable Instruments Peaks
DSP core (MIT, Copyright 2013 Emilie Gillet).
*******************************************************************************/

#ifndef DrumSynthSourceParams_H
#define DrumSynthSourceParams_H

#include <AK/SoundEngine/Common/IAkPlugin.h>
#include <AK/Plugin/PluginServices/AkFXParameterChangeHandler.h>

// These IDs map to the AudioEnginePropertyID attributes in DrumSynth.xml.
static const AkPluginParamID PARAM_MODEL_ID = 0;
static const AkPluginParamID PARAM_PARAM1_ID = 1;
static const AkPluginParamID PARAM_PARAM2_ID = 2;
static const AkPluginParamID PARAM_PARAM3_ID = 3;
static const AkPluginParamID PARAM_PARAM4_ID = 4;
static const AkPluginParamID PARAM_DURATION_ID = 5;
static const AkPluginParamID PARAM_LEVEL_ID = 6;
static const AkUInt32 NUM_PARAMS = 7;

struct DrumSynthRTPCParams
{
    // Peaks exposes four knobs whose meaning depends on the selected model.
    // For the bass drum they are frequency / punch / tone / decay.
    AkReal32 fParam1;
    AkReal32 fParam2;
    AkReal32 fParam3;
    AkReal32 fParam4;
    AkReal32 fLevel;  // dB
};

struct DrumSynthNonRTPCParams
{
    AkInt32 iModel;      // 0 = bass drum, 1 = snare, 2 = hi-hat, 3 = FM drum
    AkReal32 fDuration;  // seconds
};

struct DrumSynthSourceParams
    : public AK::IAkPluginParam
{
    DrumSynthSourceParams();
    DrumSynthSourceParams(const DrumSynthSourceParams& in_rParams);

    ~DrumSynthSourceParams();

    IAkPluginParam* Clone(AK::IAkPluginMemAlloc* in_pAllocator) override;

    AKRESULT Init(AK::IAkPluginMemAlloc* in_pAllocator, const void* in_pParamsBlock, AkUInt32 in_ulBlockSize) override;

    AKRESULT Term(AK::IAkPluginMemAlloc* in_pAllocator) override;

    AKRESULT SetParamsBlock(const void* in_pParamsBlock, AkUInt32 in_ulBlockSize) override;

    AKRESULT SetParam(AkPluginParamID in_paramID, const void* in_pValue, AkUInt32 in_ulParamSize) override;

    AK::AkFXParameterChangeHandler<NUM_PARAMS> m_paramChangeHandler;

    DrumSynthRTPCParams RTPC;
    DrumSynthNonRTPCParams NonRTPC;
};

#endif // DrumSynthSourceParams_H
