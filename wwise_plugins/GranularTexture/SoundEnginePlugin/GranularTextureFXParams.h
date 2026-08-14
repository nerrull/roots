/*******************************************************************************
Granular Texture -- a Wwise effect plug-in wrapping the Mutable Instruments
Clouds DSP core (MIT, Copyright 2014 Emilie Gillet).
*******************************************************************************/

#ifndef GranularTextureFXParams_H
#define GranularTextureFXParams_H

#include <AK/SoundEngine/Common/IAkPlugin.h>
#include <AK/Plugin/PluginServices/AkFXParameterChangeHandler.h>

// These IDs map to the AudioEnginePropertyID attributes in GranularTexture.xml.
static const AkPluginParamID PARAM_PLAYBACKMODE_ID = 0;
static const AkPluginParamID PARAM_QUALITY_ID = 1;
static const AkPluginParamID PARAM_POSITION_ID = 2;
static const AkPluginParamID PARAM_SIZE_ID = 3;
static const AkPluginParamID PARAM_PITCH_ID = 4;
static const AkPluginParamID PARAM_DENSITY_ID = 5;
static const AkPluginParamID PARAM_TEXTURE_ID = 6;
static const AkPluginParamID PARAM_DRYWET_ID = 7;
static const AkPluginParamID PARAM_STEREOSPREAD_ID = 8;
static const AkPluginParamID PARAM_FEEDBACK_ID = 9;
static const AkPluginParamID PARAM_REVERB_ID = 10;
static const AkPluginParamID PARAM_FREEZE_ID = 11;
static const AkUInt32 NUM_PARAMS = 12;

struct GranularTextureRTPCParams
{
    // Clouds' front-panel controls, all normalized 0..1 except pitch.
    AkReal32 fPosition;
    AkReal32 fSize;
    AkReal32 fPitch;         // semitones, -48..+48
    AkReal32 fDensity;
    AkReal32 fTexture;
    AkReal32 fDryWet;        // percent
    AkReal32 fStereoSpread;
    AkReal32 fFeedback;
    AkReal32 fReverb;
    bool bFreeze;            // stop writing to the buffer and play what is in it
};

struct GranularTextureNonRTPCParams
{
    AkInt32 iPlaybackMode;   // clouds::PlaybackMode
    AkInt32 iQuality;        // 0..3: stereo/mono x 16-bit/8-bit mu-law
};

struct GranularTextureFXParams
    : public AK::IAkPluginParam
{
    GranularTextureFXParams();
    GranularTextureFXParams(const GranularTextureFXParams& in_rParams);

    ~GranularTextureFXParams();

    IAkPluginParam* Clone(AK::IAkPluginMemAlloc* in_pAllocator) override;

    AKRESULT Init(AK::IAkPluginMemAlloc* in_pAllocator, const void* in_pParamsBlock, AkUInt32 in_ulBlockSize) override;

    AKRESULT Term(AK::IAkPluginMemAlloc* in_pAllocator) override;

    AKRESULT SetParamsBlock(const void* in_pParamsBlock, AkUInt32 in_ulBlockSize) override;

    AKRESULT SetParam(AkPluginParamID in_paramID, const void* in_pValue, AkUInt32 in_ulParamSize) override;

    AK::AkFXParameterChangeHandler<NUM_PARAMS> m_paramChangeHandler;

    GranularTextureRTPCParams RTPC;
    GranularTextureNonRTPCParams NonRTPC;
};

#endif // GranularTextureFXParams_H
