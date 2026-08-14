/*******************************************************************************
Modal Voice -- Wwise effect plug-in wrapping Mutable Instruments Elements.
*******************************************************************************/

#include "ModalVoiceFXParams.h"

#include <AK/Tools/Common/AkBankReadHelpers.h>

ModalVoiceFXParams::ModalVoiceFXParams()
{
}

ModalVoiceFXParams::~ModalVoiceFXParams()
{
}

ModalVoiceFXParams::ModalVoiceFXParams(const ModalVoiceFXParams& in_rParams)
{
    RTPC = in_rParams.RTPC;
    NonRTPC = in_rParams.NonRTPC;
    m_paramChangeHandler.SetAllParamChanges();
}

AK::IAkPluginParam* ModalVoiceFXParams::Clone(AK::IAkPluginMemAlloc* in_pAllocator)
{
    return AK_PLUGIN_NEW(in_pAllocator, ModalVoiceFXParams(*this));
}

AKRESULT ModalVoiceFXParams::Init(AK::IAkPluginMemAlloc* in_pAllocator, const void* in_pParamsBlock, AkUInt32 in_ulBlockSize)
{
    if (in_ulBlockSize == 0)
    {
        // A struck resonant body driven by the incoming audio: the setting that
        // makes this useful on an impact bus out of the box.
        RTPC.fPitch = 48.0f;
        RTPC.fGeometry = 0.4f;
        RTPC.fBrightness = 0.5f;
        RTPC.fDamping = 0.7f;
        RTPC.fPosition = 0.3f;
        RTPC.fSpace = 0.1f;

        RTPC.fBowLevel = 0.0f;
        RTPC.fBlowLevel = 0.0f;
        RTPC.fBlowMeta = 0.5f;
        RTPC.fStrikeLevel = 0.6f;
        RTPC.fStrikeMeta = 0.5f;
        RTPC.fContour = 0.5f;
        RTPC.fStrength = 0.7f;

        RTPC.fDryWet = 50.0f;
        RTPC.fLevel = 0.0f;
        RTPC.fSpread = 1.0f;

        NonRTPC.bGate = false;
        NonRTPC.bExternalExciter = true;

        m_paramChangeHandler.SetAllParamChanges();
        return AK_Success;
    }

    return SetParamsBlock(in_pParamsBlock, in_ulBlockSize);
}

AKRESULT ModalVoiceFXParams::Term(AK::IAkPluginMemAlloc* in_pAllocator)
{
    AK_PLUGIN_DELETE(in_pAllocator, this);
    return AK_Success;
}

AKRESULT ModalVoiceFXParams::SetParamsBlock(const void* in_pParamsBlock, AkUInt32 in_ulBlockSize)
{
    AKRESULT eResult = AK_Success;
    AkUInt8* pParamsBlock = (AkUInt8*)in_pParamsBlock;

    RTPC.fPitch = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fGeometry = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fBrightness = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fDamping = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fPosition = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fSpace = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fBowLevel = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fBlowLevel = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fBlowMeta = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fStrikeLevel = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fStrikeMeta = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fContour = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fStrength = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fDryWet = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fLevel = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fSpread = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    NonRTPC.bGate = READBANKDATA(bool, pParamsBlock, in_ulBlockSize);
    NonRTPC.bExternalExciter = READBANKDATA(bool, pParamsBlock, in_ulBlockSize);
    CHECKBANKDATASIZE(in_ulBlockSize, eResult);
    m_paramChangeHandler.SetAllParamChanges();

    return eResult;
}

AKRESULT ModalVoiceFXParams::SetParam(AkPluginParamID in_paramID, const void* in_pValue, AkUInt32 in_ulParamSize)
{
    AKRESULT eResult = AK_Success;

    switch (in_paramID)
    {
    case PARAM_PITCH_ID:        RTPC.fPitch = *((AkReal32*)in_pValue); break;
    case PARAM_GEOMETRY_ID:     RTPC.fGeometry = *((AkReal32*)in_pValue); break;
    case PARAM_BRIGHTNESS_ID:   RTPC.fBrightness = *((AkReal32*)in_pValue); break;
    case PARAM_DAMPING_ID:      RTPC.fDamping = *((AkReal32*)in_pValue); break;
    case PARAM_POSITION_ID:     RTPC.fPosition = *((AkReal32*)in_pValue); break;
    case PARAM_SPACE_ID:        RTPC.fSpace = *((AkReal32*)in_pValue); break;
    case PARAM_BOWLEVEL_ID:     RTPC.fBowLevel = *((AkReal32*)in_pValue); break;
    case PARAM_BLOWLEVEL_ID:    RTPC.fBlowLevel = *((AkReal32*)in_pValue); break;
    case PARAM_BLOWMETA_ID:     RTPC.fBlowMeta = *((AkReal32*)in_pValue); break;
    case PARAM_STRIKELEVEL_ID:  RTPC.fStrikeLevel = *((AkReal32*)in_pValue); break;
    case PARAM_STRIKEMETA_ID:   RTPC.fStrikeMeta = *((AkReal32*)in_pValue); break;
    case PARAM_CONTOUR_ID:      RTPC.fContour = *((AkReal32*)in_pValue); break;
    case PARAM_STRENGTH_ID:     RTPC.fStrength = *((AkReal32*)in_pValue); break;
    case PARAM_DRYWET_ID:       RTPC.fDryWet = *((AkReal32*)in_pValue); break;
    case PARAM_LEVEL_ID:        RTPC.fLevel = *((AkReal32*)in_pValue); break;
    case PARAM_SPREAD_ID:       RTPC.fSpread = *((AkReal32*)in_pValue); break;
    case PARAM_GATE_ID:         NonRTPC.bGate = *((bool*)in_pValue); break;
    case PARAM_EXTERNALEXCITER_ID: NonRTPC.bExternalExciter = *((bool*)in_pValue); break;
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
