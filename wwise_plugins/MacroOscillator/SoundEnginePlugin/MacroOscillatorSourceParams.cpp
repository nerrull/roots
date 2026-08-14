/*******************************************************************************
Macro Oscillator -- Wwise source plug-in wrapping Mutable Instruments Plaits.
*******************************************************************************/

#include "MacroOscillatorSourceParams.h"

#include <AK/Tools/Common/AkBankReadHelpers.h>

MacroOscillatorSourceParams::MacroOscillatorSourceParams()
{
}

MacroOscillatorSourceParams::~MacroOscillatorSourceParams()
{
}

MacroOscillatorSourceParams::MacroOscillatorSourceParams(const MacroOscillatorSourceParams& in_rParams)
{
    RTPC = in_rParams.RTPC;
    NonRTPC = in_rParams.NonRTPC;
    m_paramChangeHandler.SetAllParamChanges();
}

AK::IAkPluginParam* MacroOscillatorSourceParams::Clone(AK::IAkPluginMemAlloc* in_pAllocator)
{
    return AK_PLUGIN_NEW(in_pAllocator, MacroOscillatorSourceParams(*this));
}

AKRESULT MacroOscillatorSourceParams::Init(AK::IAkPluginMemAlloc* in_pAllocator, const void* in_pParamsBlock, AkUInt32 in_ulBlockSize)
{
    if (in_ulBlockSize == 0)
    {
        // Engine 0 is the virtual-analog pair; a plucked 1 s note is a sane
        // thing to hear the first time the plug-in is dropped on a sound.
        RTPC.fPitch = 60.0f;
        RTPC.fHarmonics = 0.5f;
        RTPC.fTimbre = 0.5f;
        RTPC.fMorph = 0.5f;
        RTPC.fDecay = 0.5f;
        RTPC.fLpgColour = 0.5f;
        RTPC.fAuxMix = 0.0f;
        RTPC.fLevel = 0.0f;

        NonRTPC.iEngine = 0;
        NonRTPC.fDuration = 1.0f;
        NonRTPC.bTriggered = true;

        m_paramChangeHandler.SetAllParamChanges();
        return AK_Success;
    }

    return SetParamsBlock(in_pParamsBlock, in_ulBlockSize);
}

AKRESULT MacroOscillatorSourceParams::Term(AK::IAkPluginMemAlloc* in_pAllocator)
{
    AK_PLUGIN_DELETE(in_pAllocator, this);
    return AK_Success;
}

AKRESULT MacroOscillatorSourceParams::SetParamsBlock(const void* in_pParamsBlock, AkUInt32 in_ulBlockSize)
{
    AKRESULT eResult = AK_Success;
    AkUInt8* pParamsBlock = (AkUInt8*)in_pParamsBlock;

    // Must match the order GetBankParameters() writes them in.
    NonRTPC.iEngine = READBANKDATA(AkInt32, pParamsBlock, in_ulBlockSize);
    RTPC.fPitch = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fHarmonics = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fTimbre = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fMorph = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fDecay = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fLpgColour = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fAuxMix = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    NonRTPC.fDuration = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    RTPC.fLevel = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    NonRTPC.bTriggered = READBANKDATA(bool, pParamsBlock, in_ulBlockSize);
    CHECKBANKDATASIZE(in_ulBlockSize, eResult);
    m_paramChangeHandler.SetAllParamChanges();

    return eResult;
}

AKRESULT MacroOscillatorSourceParams::SetParam(AkPluginParamID in_paramID, const void* in_pValue, AkUInt32 in_ulParamSize)
{
    AKRESULT eResult = AK_Success;

    switch (in_paramID)
    {
    case PARAM_ENGINE_ID:
        NonRTPC.iEngine = *((AkInt32*)in_pValue);
        break;
    case PARAM_PITCH_ID:
        RTPC.fPitch = *((AkReal32*)in_pValue);
        break;
    case PARAM_HARMONICS_ID:
        RTPC.fHarmonics = *((AkReal32*)in_pValue);
        break;
    case PARAM_TIMBRE_ID:
        RTPC.fTimbre = *((AkReal32*)in_pValue);
        break;
    case PARAM_MORPH_ID:
        RTPC.fMorph = *((AkReal32*)in_pValue);
        break;
    case PARAM_DECAY_ID:
        RTPC.fDecay = *((AkReal32*)in_pValue);
        break;
    case PARAM_LPGCOLOUR_ID:
        RTPC.fLpgColour = *((AkReal32*)in_pValue);
        break;
    case PARAM_AUXMIX_ID:
        RTPC.fAuxMix = *((AkReal32*)in_pValue);
        break;
    case PARAM_DURATION_ID:
        NonRTPC.fDuration = *((AkReal32*)in_pValue);
        break;
    case PARAM_LEVEL_ID:
        RTPC.fLevel = *((AkReal32*)in_pValue);
        break;
    case PARAM_TRIGGERED_ID:
        NonRTPC.bTriggered = *((bool*)in_pValue);
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
