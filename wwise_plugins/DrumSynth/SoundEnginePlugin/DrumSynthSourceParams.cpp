/*******************************************************************************
Drum Synth -- Wwise source plug-in wrapping Mutable Instruments Peaks.
*******************************************************************************/

#include "DrumSynthSourceParams.h"

#include <AK/Tools/Common/AkBankReadHelpers.h>

DrumSynthSourceParams::DrumSynthSourceParams()
{
}

DrumSynthSourceParams::~DrumSynthSourceParams()
{
}

DrumSynthSourceParams::DrumSynthSourceParams(const DrumSynthSourceParams& in_rParams)
{
    RTPC = in_rParams.RTPC;
    NonRTPC = in_rParams.NonRTPC;
    m_paramChangeHandler.SetAllParamChanges();
}

AK::IAkPluginParam* DrumSynthSourceParams::Clone(AK::IAkPluginMemAlloc* in_pAllocator)
{
    return AK_PLUGIN_NEW(in_pAllocator, DrumSynthSourceParams(*this));
}

AKRESULT DrumSynthSourceParams::Init(AK::IAkPluginMemAlloc* in_pAllocator, const void* in_pParamsBlock, AkUInt32 in_ulBlockSize)
{
    if (in_ulBlockSize == 0)
    {
        RTPC.fParam1 = 0.3f;
        RTPC.fParam2 = 0.5f;
        RTPC.fParam3 = 0.5f;
        RTPC.fParam4 = 0.5f;
        RTPC.fLevel = 0.0f;

        NonRTPC.iModel = 0;  // bass drum
        NonRTPC.fDuration = 1.0f;

        m_paramChangeHandler.SetAllParamChanges();
        return AK_Success;
    }

    return SetParamsBlock(in_pParamsBlock, in_ulBlockSize);
}

AKRESULT DrumSynthSourceParams::Term(AK::IAkPluginMemAlloc* in_pAllocator)
{
    AK_PLUGIN_DELETE(in_pAllocator, this);
    return AK_Success;
}

AKRESULT DrumSynthSourceParams::SetParamsBlock(const void* in_pParamsBlock, AkUInt32 in_ulBlockSize)
{
    AKRESULT eResult = AK_Success;
    AkUInt8* pParamsBlock = (AkUInt8*)in_pParamsBlock;

    NonRTPC.iModel = READBANKDATA(AkInt32, pParamsBlock, in_ulBlockSize);
    RTPC.fParam1 = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fParam2 = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fParam3 = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fParam4 = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    NonRTPC.fDuration = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fLevel = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    CHECKBANKDATASIZE(in_ulBlockSize, eResult);
    m_paramChangeHandler.SetAllParamChanges();

    return eResult;
}

AKRESULT DrumSynthSourceParams::SetParam(AkPluginParamID in_paramID, const void* in_pValue, AkUInt32 in_ulParamSize)
{
    AKRESULT eResult = AK_Success;

    switch (in_paramID)
    {
    case PARAM_MODEL_ID:
        NonRTPC.iModel = *((AkInt32*)in_pValue);
        break;
    case PARAM_PARAM1_ID:
        RTPC.fParam1 = *((AkReal32*)in_pValue);
        break;
    case PARAM_PARAM2_ID:
        RTPC.fParam2 = *((AkReal32*)in_pValue);
        break;
    case PARAM_PARAM3_ID:
        RTPC.fParam3 = *((AkReal32*)in_pValue);
        break;
    case PARAM_PARAM4_ID:
        RTPC.fParam4 = *((AkReal32*)in_pValue);
        break;
    case PARAM_DURATION_ID:
        NonRTPC.fDuration = *((AkReal32*)in_pValue);
        break;
    case PARAM_LEVEL_ID:
        RTPC.fLevel = *((AkReal32*)in_pValue);
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
