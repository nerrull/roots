/*******************************************************************************
Modal Voice -- a Wwise effect plug-in wrapping the Mutable Instruments Elements
DSP core (MIT, Copyright 2014 Emilie Gillet).
*******************************************************************************/

#ifndef ModalVoiceFXParams_H
#define ModalVoiceFXParams_H

#include <AK/SoundEngine/Common/IAkPlugin.h>
#include <AK/Plugin/PluginServices/AkFXParameterChangeHandler.h>

// These IDs map to the AudioEnginePropertyID attributes in ModalVoice.xml.
static const AkPluginParamID PARAM_PITCH_ID = 0;
static const AkPluginParamID PARAM_GEOMETRY_ID = 1;
static const AkPluginParamID PARAM_BRIGHTNESS_ID = 2;
static const AkPluginParamID PARAM_DAMPING_ID = 3;
static const AkPluginParamID PARAM_POSITION_ID = 4;
static const AkPluginParamID PARAM_SPACE_ID = 5;
static const AkPluginParamID PARAM_BOWLEVEL_ID = 6;
static const AkPluginParamID PARAM_BLOWLEVEL_ID = 7;
static const AkPluginParamID PARAM_BLOWMETA_ID = 8;
static const AkPluginParamID PARAM_STRIKELEVEL_ID = 9;
static const AkPluginParamID PARAM_STRIKEMETA_ID = 10;
static const AkPluginParamID PARAM_CONTOUR_ID = 11;
static const AkPluginParamID PARAM_STRENGTH_ID = 12;
static const AkPluginParamID PARAM_DRYWET_ID = 13;
static const AkPluginParamID PARAM_LEVEL_ID = 14;
static const AkPluginParamID PARAM_SPREAD_ID = 15;
static const AkPluginParamID PARAM_GATE_ID = 16;
static const AkPluginParamID PARAM_EXTERNALEXCITER_ID = 17;
static const AkUInt32 NUM_PARAMS = 18;

struct ModalVoiceRTPCParams
{
    AkReal32 fPitch;        // MIDI note number

    // Resonator.
    AkReal32 fGeometry;
    AkReal32 fBrightness;
    AkReal32 fDamping;
    AkReal32 fPosition;
    AkReal32 fSpace;        // built-in reverb amount

    // Exciter section.
    AkReal32 fBowLevel;
    AkReal32 fBlowLevel;
    AkReal32 fBlowMeta;
    AkReal32 fStrikeLevel;
    AkReal32 fStrikeMeta;
    AkReal32 fContour;      // exciter envelope shape
    AkReal32 fStrength;     // accent/velocity

    AkReal32 fDryWet;       // percent
    AkReal32 fLevel;        // dB
    AkReal32 fSpread;       // 0 = mono sum of main/aux, 1 = fully separated
};

struct ModalVoiceNonRTPCParams
{
    bool bGate;             // hold the exciter envelope open
    bool bExternalExciter;  // feed the bus audio into the exciter inputs
};

struct ModalVoiceFXParams
    : public AK::IAkPluginParam
{
    ModalVoiceFXParams();
    ModalVoiceFXParams(const ModalVoiceFXParams& in_rParams);

    ~ModalVoiceFXParams();

    IAkPluginParam* Clone(AK::IAkPluginMemAlloc* in_pAllocator) override;

    AKRESULT Init(AK::IAkPluginMemAlloc* in_pAllocator, const void* in_pParamsBlock, AkUInt32 in_ulBlockSize) override;

    AKRESULT Term(AK::IAkPluginMemAlloc* in_pAllocator) override;

    AKRESULT SetParamsBlock(const void* in_pParamsBlock, AkUInt32 in_ulBlockSize) override;

    AKRESULT SetParam(AkPluginParamID in_paramID, const void* in_pValue, AkUInt32 in_ulParamSize) override;

    AK::AkFXParameterChangeHandler<NUM_PARAMS> m_paramChangeHandler;

    ModalVoiceRTPCParams RTPC;
    ModalVoiceNonRTPCParams NonRTPC;
};

#endif // ModalVoiceFXParams_H
