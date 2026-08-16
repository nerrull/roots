/*******************************************************************************
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
*******************************************************************************/

#include "OnsetTapFXParams.h"

#include <AK/Tools/Common/AkBankReadHelpers.h>

#include <cstring>

OnsetTapFXParams::OnsetTapFXParams()
{
}

OnsetTapFXParams::~OnsetTapFXParams()
{
}

OnsetTapFXParams::OnsetTapFXParams(const OnsetTapFXParams& in_rParams)
{
    NonRTPC = in_rParams.NonRTPC;
    m_paramChangeHandler.SetAllParamChanges();
}

AK::IAkPluginParam* OnsetTapFXParams::Clone(AK::IAkPluginMemAlloc* in_pAllocator)
{
    return AK_PLUGIN_NEW(in_pAllocator, OnsetTapFXParams(*this));
}

AKRESULT OnsetTapFXParams::Init(AK::IAkPluginMemAlloc* in_pAllocator, const void* in_pParamsBlock, AkUInt32 in_ulBlockSize)
{
    if (in_ulBlockSize == 0)
    {
        NonRTPC.iTapId = 0;
        NonRTPC.bEnabled = true;
        NonRTPC.szName[0] = '\0';
        NonRTPC.fSensitivity = 2.5f;
        NonRTPC.fFloorDb = -60.f;
        NonRTPC.fMinRiseDb = 2.f;
        NonRTPC.fMinIntervalMs = 80.f;
        NonRTPC.fWindowMs = 1000.f;
        m_paramChangeHandler.SetAllParamChanges();
        return AK_Success;
    }

    return SetParamsBlock(in_pParamsBlock, in_ulBlockSize);
}

AKRESULT OnsetTapFXParams::Term(AK::IAkPluginMemAlloc* in_pAllocator)
{
    AK_PLUGIN_DELETE(in_pAllocator, this);
    return AK_Success;
}

AKRESULT OnsetTapFXParams::SetParamsBlock(const void* in_pParamsBlock, AkUInt32 in_ulBlockSize)
{
    AKRESULT eResult = AK_Success;
    AkUInt8* pParamsBlock = (AkUInt8*)in_pParamsBlock;

    // Order must match OnsetTap.xml's AudioEnginePropertyID order and
    // OnsetTapPlugin::GetBankParameters.
    NonRTPC.iTapId = READBANKDATA(AkInt32, pParamsBlock, in_ulBlockSize);
    NonRTPC.bEnabled = READBANKDATA(bool, pParamsBlock, in_ulBlockSize);
    COPYBANKSTRING_CHAR(pParamsBlock, in_ulBlockSize, NonRTPC.szName, kMaxNameLen);
    NonRTPC.fSensitivity = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    NonRTPC.fFloorDb = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    NonRTPC.fMinRiseDb = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    NonRTPC.fMinIntervalMs = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    NonRTPC.fWindowMs = READBANKDATA(AkReal32, pParamsBlock, in_ulBlockSize);
    CHECKBANKDATASIZE(in_ulBlockSize, eResult);
    m_paramChangeHandler.SetAllParamChanges();

    return eResult;
}

AKRESULT OnsetTapFXParams::SetParam(AkPluginParamID in_paramID, const void* in_pValue, AkUInt32 in_ulParamSize)
{
    AKRESULT eResult = AK_Success;

    switch (in_paramID)
    {
    case PARAM_TAP_ID_ID:
        NonRTPC.iTapId = *((AkInt32*)in_pValue);
        break;
    case PARAM_ENABLED_ID:
        NonRTPC.bEnabled = *((bool*)in_pValue);
        break;
    case PARAM_NAME_ID:
    {
        const AkUInt32 uCopyLen = in_ulParamSize < (kMaxNameLen - 1) ? in_ulParamSize : (kMaxNameLen - 1);
        memcpy(NonRTPC.szName, in_pValue, uCopyLen);
        NonRTPC.szName[uCopyLen] = '\0';
        break;
    }
    case PARAM_SENSITIVITY_ID:
        NonRTPC.fSensitivity = *((AkReal32*)in_pValue);
        break;
    case PARAM_FLOOR_DB_ID:
        NonRTPC.fFloorDb = *((AkReal32*)in_pValue);
        break;
    case PARAM_MIN_RISE_DB_ID:
        NonRTPC.fMinRiseDb = *((AkReal32*)in_pValue);
        break;
    case PARAM_MIN_INTERVAL_MS_ID:
        NonRTPC.fMinIntervalMs = *((AkReal32*)in_pValue);
        break;
    case PARAM_WINDOW_MS_ID:
        NonRTPC.fWindowMs = *((AkReal32*)in_pValue);
        break;
    default:
        eResult = AK_InvalidParameter;
        break;
    }

    if (eResult == AK_Success)
        m_paramChangeHandler.SetParamChange(in_paramID);

    return eResult;
}
