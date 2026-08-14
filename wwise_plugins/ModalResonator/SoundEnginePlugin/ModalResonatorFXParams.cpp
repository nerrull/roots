/*******************************************************************************
Modal Resonator -- Wwise effect plug-in wrapping Mutable Instruments Rings.
*******************************************************************************/

#include "ModalResonatorFXParams.h"

#include <AK/Tools/Common/AkBankReadHelpers.h>

ModalResonatorFXParams::ModalResonatorFXParams()
{
}

ModalResonatorFXParams::~ModalResonatorFXParams()
{
}

ModalResonatorFXParams::ModalResonatorFXParams(const ModalResonatorFXParams& in_rParams)
{
    RTPC = in_rParams.RTPC;
    NonRTPC = in_rParams.NonRTPC;
    m_paramChangeHandler.SetAllParamChanges();
}

AK::IAkPluginParam* ModalResonatorFXParams::Clone(AK::IAkPluginMemAlloc* in_pAllocator)
{
    return AK_PLUGIN_NEW(in_pAllocator, ModalResonatorFXParams(*this));
}

AKRESULT ModalResonatorFXParams::Init(AK::IAkPluginMemAlloc* in_pAllocator, const void* in_pParamsBlock, AkUInt32 in_ulBlockSize)
{
    if (in_ulBlockSize == 0)
    {
        // Defaults chosen to be immediately useful on an impact/percussive
        // source: a struck modal body, medium decay, driven by the input.
        RTPC.fStructure = 0.25f;
        RTPC.fBrightness = 0.5f;
        RTPC.fDamping = 0.7f;
        RTPC.fPosition = 0.25f;
        RTPC.fPitch = 60.0f;
        RTPC.fSpread = 1.0f;
        RTPC.fDryWet = 50.0f;
        RTPC.fOutputLevel = 0.0f;

        NonRTPC.iModel = 0;  // RESONATOR_MODEL_MODAL
        NonRTPC.iChord = 0;
        NonRTPC.iPolyphony = 1;
        NonRTPC.bInternalExciter = false;

        m_paramChangeHandler.SetAllParamChanges();
        return AK_Success;
    }

    return SetParamsBlock(in_pParamsBlock, in_ulBlockSize);
}

AKRESULT ModalResonatorFXParams::Term(AK::IAkPluginMemAlloc* in_pAllocator)
{
    AK_PLUGIN_DELETE(in_pAllocator, this);
    return AK_Success;
}

AKRESULT ModalResonatorFXParams::SetParamsBlock(const void* in_pParamsBlock, AkUInt32 in_ulBlockSize)
{
    AKRESULT eResult = AK_Success;
    AkUInt8* pParamsBlock = (AkUInt8*)in_pParamsBlock;

    // This must stay in the same order as GetBankParameters() in the authoring
    // plug-in writes them.
    NonRTPC.iModel = READBANKDATA(AkInt32, pParamsBlock, in_ulBlockSize);
    RTPC.fStructure = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fBrightness = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fDamping = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fPosition = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fPitch = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    NonRTPC.iChord = READBANKDATA(AkInt32, pParamsBlock, in_ulBlockSize);
    NonRTPC.iPolyphony = READBANKDATA(AkInt32, pParamsBlock, in_ulBlockSize);
    NonRTPC.bInternalExciter = READBANKDATA(bool, pParamsBlock, in_ulBlockSize);
    RTPC.fSpread = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fDryWet = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fOutputLevel = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    CHECKBANKDATASIZE(in_ulBlockSize, eResult);
    m_paramChangeHandler.SetAllParamChanges();

    return eResult;
}

AKRESULT ModalResonatorFXParams::SetParam(AkPluginParamID in_paramID, const void* in_pValue, AkUInt32 in_ulParamSize)
{
    AKRESULT eResult = AK_Success;

    switch (in_paramID)
    {
    case PARAM_MODEL_ID:
        NonRTPC.iModel = *((AkInt32*)in_pValue);
        break;
    case PARAM_STRUCTURE_ID:
        RTPC.fStructure = *((AkReal32*)in_pValue);
        break;
    case PARAM_BRIGHTNESS_ID:
        RTPC.fBrightness = *((AkReal32*)in_pValue);
        break;
    case PARAM_DAMPING_ID:
        RTPC.fDamping = *((AkReal32*)in_pValue);
        break;
    case PARAM_POSITION_ID:
        RTPC.fPosition = *((AkReal32*)in_pValue);
        break;
    case PARAM_PITCH_ID:
        RTPC.fPitch = *((AkReal32*)in_pValue);
        break;
    case PARAM_CHORD_ID:
        NonRTPC.iChord = *((AkInt32*)in_pValue);
        break;
    case PARAM_POLYPHONY_ID:
        NonRTPC.iPolyphony = *((AkInt32*)in_pValue);
        break;
    case PARAM_INTERNALEXCITER_ID:
        NonRTPC.bInternalExciter = *((bool*)in_pValue);
        break;
    case PARAM_SPREAD_ID:
        RTPC.fSpread = *((AkReal32*)in_pValue);
        break;
    case PARAM_DRYWET_ID:
        RTPC.fDryWet = *((AkReal32*)in_pValue);
        break;
    case PARAM_OUTPUTLEVEL_ID:
        RTPC.fOutputLevel = *((AkReal32*)in_pValue);
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
