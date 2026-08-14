/*******************************************************************************
Modal Resonator -- a Wwise effect plug-in wrapping the Mutable Instruments
Rings DSP core (MIT, Copyright 2015 Emilie Gillet).

The plug-in is built from the unmodified sources in the eurorack repository;
everything specific to Wwise lives in this project.
*******************************************************************************/

#ifndef ModalResonatorFXParams_H
#define ModalResonatorFXParams_H

#include <AK/SoundEngine/Common/IAkPlugin.h>
#include <AK/Plugin/PluginServices/AkFXParameterChangeHandler.h>

// These IDs map to the AudioEnginePropertyID attributes in ModalResonator.xml.
static const AkPluginParamID PARAM_MODEL_ID = 0;
static const AkPluginParamID PARAM_STRUCTURE_ID = 1;
static const AkPluginParamID PARAM_BRIGHTNESS_ID = 2;
static const AkPluginParamID PARAM_DAMPING_ID = 3;
static const AkPluginParamID PARAM_POSITION_ID = 4;
static const AkPluginParamID PARAM_PITCH_ID = 5;
static const AkPluginParamID PARAM_CHORD_ID = 6;
static const AkPluginParamID PARAM_POLYPHONY_ID = 7;
static const AkPluginParamID PARAM_INTERNALEXCITER_ID = 8;
static const AkPluginParamID PARAM_SPREAD_ID = 9;
static const AkPluginParamID PARAM_DRYWET_ID = 10;
static const AkPluginParamID PARAM_OUTPUTLEVEL_ID = 11;
static const AkUInt32 NUM_PARAMS = 12;

struct ModalResonatorRTPCParams
{
    // Rings' four macro controls, all normalized 0..1 exactly as the module's
    // front-panel knobs are.
    AkReal32 fStructure;
    AkReal32 fBrightness;
    AkReal32 fDamping;
    AkReal32 fPosition;

    // Pitch of the resonator, as a MIDI note number.
    AkReal32 fPitch;

    // 0 = mono sum of both resonator outputs on every channel,
    // 1 = 'out' and 'aux' fully separated across the stereo field.
    AkReal32 fSpread;

    AkReal32 fDryWet;       // percent
    AkReal32 fOutputLevel;  // dB
};

struct ModalResonatorNonRTPCParams
{
    AkInt32 iModel;       // rings::ResonatorModel
    AkInt32 iChord;       // 0..10
    AkInt32 iPolyphony;   // 1, 2 or 4
    bool bInternalExciter;
};

struct ModalResonatorFXParams
    : public AK::IAkPluginParam
{
    ModalResonatorFXParams();
    ModalResonatorFXParams(const ModalResonatorFXParams& in_rParams);

    ~ModalResonatorFXParams();

    /// Create a duplicate of the parameter node instance in its current state.
    IAkPluginParam* Clone(AK::IAkPluginMemAlloc* in_pAllocator) override;

    /// Initialize the plug-in parameter node interface.
    /// Initializes the internal parameter structure to default values or with the provided parameter block if it is valid.
    AKRESULT Init(AK::IAkPluginMemAlloc* in_pAllocator, const void* in_pParamsBlock, AkUInt32 in_ulBlockSize) override;

    /// Called by the sound engine when a parameter node is terminated.
    AKRESULT Term(AK::IAkPluginMemAlloc* in_pAllocator) override;

    /// Set all plug-in parameters at once using a parameter block.
    AKRESULT SetParamsBlock(const void* in_pParamsBlock, AkUInt32 in_ulBlockSize) override;

    /// Update a single parameter at a time and perform the necessary actions on the parameter changes.
    AKRESULT SetParam(AkPluginParamID in_paramID, const void* in_pValue, AkUInt32 in_ulParamSize) override;

    AK::AkFXParameterChangeHandler<NUM_PARAMS> m_paramChangeHandler;

    ModalResonatorRTPCParams RTPC;
    ModalResonatorNonRTPCParams NonRTPC;
};

#endif // ModalResonatorFXParams_H
