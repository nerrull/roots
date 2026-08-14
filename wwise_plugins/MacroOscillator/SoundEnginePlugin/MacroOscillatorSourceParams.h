/*******************************************************************************
Macro Oscillator -- a Wwise source plug-in wrapping the Mutable Instruments
Plaits DSP core (MIT, Copyright 2016 Emilie Gillet).
*******************************************************************************/

#ifndef MacroOscillatorSourceParams_H
#define MacroOscillatorSourceParams_H

#include <AK/SoundEngine/Common/IAkPlugin.h>
#include <AK/Plugin/PluginServices/AkFXParameterChangeHandler.h>

// These IDs map to the AudioEnginePropertyID attributes in MacroOscillator.xml.
static const AkPluginParamID PARAM_ENGINE_ID = 0;
static const AkPluginParamID PARAM_PITCH_ID = 1;
static const AkPluginParamID PARAM_HARMONICS_ID = 2;
static const AkPluginParamID PARAM_TIMBRE_ID = 3;
static const AkPluginParamID PARAM_MORPH_ID = 4;
static const AkPluginParamID PARAM_DECAY_ID = 5;
static const AkPluginParamID PARAM_LPGCOLOUR_ID = 6;
static const AkPluginParamID PARAM_AUXMIX_ID = 7;
static const AkPluginParamID PARAM_DURATION_ID = 8;
static const AkPluginParamID PARAM_LEVEL_ID = 9;
static const AkPluginParamID PARAM_TRIGGERED_ID = 10;
static const AkUInt32 NUM_PARAMS = 11;

struct MacroOscillatorRTPCParams
{
    AkReal32 fPitch;      // MIDI note number
    AkReal32 fHarmonics;  // 0..1, Plaits' HARMONICS knob
    AkReal32 fTimbre;     // 0..1
    AkReal32 fMorph;      // 0..1
    AkReal32 fDecay;      // 0..1, low-pass gate decay
    AkReal32 fLpgColour;  // 0..1, VCA..VCF character of the gate
    AkReal32 fAuxMix;     // 0 = main output only, 1 = aux output only
    AkReal32 fLevel;      // dB
};

struct MacroOscillatorNonRTPCParams
{
    AkInt32 iEngine;    // 0..23, selects the synthesis model
    AkReal32 fDuration; // seconds; 0 means play until stopped
    bool bTriggered;    // fire the low-pass gate envelope when the voice starts
};

struct MacroOscillatorSourceParams
    : public AK::IAkPluginParam
{
    MacroOscillatorSourceParams();
    MacroOscillatorSourceParams(const MacroOscillatorSourceParams& in_rParams);

    ~MacroOscillatorSourceParams();

    IAkPluginParam* Clone(AK::IAkPluginMemAlloc* in_pAllocator) override;

    AKRESULT Init(AK::IAkPluginMemAlloc* in_pAllocator, const void* in_pParamsBlock, AkUInt32 in_ulBlockSize) override;

    AKRESULT Term(AK::IAkPluginMemAlloc* in_pAllocator) override;

    AKRESULT SetParamsBlock(const void* in_pParamsBlock, AkUInt32 in_ulBlockSize) override;

    AKRESULT SetParam(AkPluginParamID in_paramID, const void* in_pValue, AkUInt32 in_ulParamSize) override;

    AK::AkFXParameterChangeHandler<NUM_PARAMS> m_paramChangeHandler;

    MacroOscillatorRTPCParams RTPC;
    MacroOscillatorNonRTPCParams NonRTPC;
};

#endif // MacroOscillatorSourceParams_H
