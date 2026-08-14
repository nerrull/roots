/*******************************************************************************
Granular Texture -- Wwise effect plug-in wrapping Mutable Instruments Clouds.
*******************************************************************************/

#include "GranularTextureFXParams.h"

#include <AK/Tools/Common/AkBankReadHelpers.h>

GranularTextureFXParams::GranularTextureFXParams()
{
}

GranularTextureFXParams::~GranularTextureFXParams()
{
}

GranularTextureFXParams::GranularTextureFXParams(const GranularTextureFXParams& in_rParams)
{
    RTPC = in_rParams.RTPC;
    NonRTPC = in_rParams.NonRTPC;
    m_paramChangeHandler.SetAllParamChanges();
}

AK::IAkPluginParam* GranularTextureFXParams::Clone(AK::IAkPluginMemAlloc* in_pAllocator)
{
    return AK_PLUGIN_NEW(in_pAllocator, GranularTextureFXParams(*this));
}

AKRESULT GranularTextureFXParams::Init(AK::IAkPluginMemAlloc* in_pAllocator, const void* in_pParamsBlock, AkUInt32 in_ulBlockSize)
{
    if (in_ulBlockSize == 0)
    {
        // Mid-position grains of medium size at moderate density: the setting
        // that makes an incoming sound audibly granular without destroying it.
        RTPC.fPosition = 0.5f;
        RTPC.fSize = 0.5f;
        RTPC.fPitch = 0.0f;
        RTPC.fDensity = 0.5f;
        RTPC.fTexture = 0.5f;
        RTPC.fDryWet = 50.0f;
        RTPC.fStereoSpread = 0.5f;
        RTPC.fFeedback = 0.0f;
        RTPC.fReverb = 0.0f;
        RTPC.bFreeze = false;

        NonRTPC.iPlaybackMode = 0;  // PLAYBACK_MODE_GRANULAR
        NonRTPC.iQuality = 0;       // stereo, 16-bit

        m_paramChangeHandler.SetAllParamChanges();
        return AK_Success;
    }

    return SetParamsBlock(in_pParamsBlock, in_ulBlockSize);
}

AKRESULT GranularTextureFXParams::Term(AK::IAkPluginMemAlloc* in_pAllocator)
{
    AK_PLUGIN_DELETE(in_pAllocator, this);
    return AK_Success;
}

AKRESULT GranularTextureFXParams::SetParamsBlock(const void* in_pParamsBlock, AkUInt32 in_ulBlockSize)
{
    AKRESULT eResult = AK_Success;
    AkUInt8* pParamsBlock = (AkUInt8*)in_pParamsBlock;

    NonRTPC.iPlaybackMode = READBANKDATA(AkInt32, pParamsBlock, in_ulBlockSize);
    NonRTPC.iQuality = READBANKDATA(AkInt32, pParamsBlock, in_ulBlockSize);
    RTPC.fPosition = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fSize = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fPitch = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fDensity = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fTexture = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fDryWet = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fStereoSpread = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fFeedback = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fReverb = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.bFreeze = READBANKDATA(bool, pParamsBlock, in_ulBlockSize);
    CHECKBANKDATASIZE(in_ulBlockSize, eResult);
    m_paramChangeHandler.SetAllParamChanges();

    return eResult;
}

AKRESULT GranularTextureFXParams::SetParam(AkPluginParamID in_paramID, const void* in_pValue, AkUInt32 in_ulParamSize)
{
    AKRESULT eResult = AK_Success;

    switch (in_paramID)
    {
    case PARAM_PLAYBACKMODE_ID:
        NonRTPC.iPlaybackMode = *((AkInt32*)in_pValue);
        break;
    case PARAM_QUALITY_ID:
        NonRTPC.iQuality = *((AkInt32*)in_pValue);
        break;
    case PARAM_POSITION_ID:
        RTPC.fPosition = *((AkReal32*)in_pValue);
        break;
    case PARAM_SIZE_ID:
        RTPC.fSize = *((AkReal32*)in_pValue);
        break;
    case PARAM_PITCH_ID:
        RTPC.fPitch = *((AkReal32*)in_pValue);
        break;
    case PARAM_DENSITY_ID:
        RTPC.fDensity = *((AkReal32*)in_pValue);
        break;
    case PARAM_TEXTURE_ID:
        RTPC.fTexture = *((AkReal32*)in_pValue);
        break;
    case PARAM_DRYWET_ID:
        RTPC.fDryWet = *((AkReal32*)in_pValue);
        break;
    case PARAM_STEREOSPREAD_ID:
        RTPC.fStereoSpread = *((AkReal32*)in_pValue);
        break;
    case PARAM_FEEDBACK_ID:
        RTPC.fFeedback = *((AkReal32*)in_pValue);
        break;
    case PARAM_REVERB_ID:
        RTPC.fReverb = *((AkReal32*)in_pValue);
        break;
    case PARAM_FREEZE_ID:
        RTPC.bFreeze = *((bool*)in_pValue);
        break;
    default:
        eResult = AK_InvalidParameter;
        break;
    }

    if (eResult == AK_Success)
    {
        m_paramChangeHandler.SetParamChange(in_paramID);
    }

    return eResult;
}
