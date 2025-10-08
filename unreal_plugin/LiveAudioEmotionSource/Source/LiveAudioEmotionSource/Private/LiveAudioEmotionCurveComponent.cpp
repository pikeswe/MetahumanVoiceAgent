#include "LiveAudioEmotionCurveComponent.h"

#include "LiveAudioEmotionSourceSubsystem.h"
#include "Sound/SoundWaveProcedural.h"

ULiveAudioEmotionCurveComponent::ULiveAudioEmotionCurveComponent()
{
    PrimaryComponentTick.bCanEverTick = true;
    CurveSourceBindingName = TEXT("MVA_Voice");
    CurveSyncOffset = 0.f;
}

void ULiveAudioEmotionCurveComponent::BeginPlay()
{
    Super::BeginPlay();
    if (UGameInstance* GameInstance = GetWorld()->GetGameInstance())
    {
        Subsystem = GameInstance->GetSubsystem<ULiveAudioEmotionSourceSubsystem>();
        if (Subsystem.IsValid())
        {
            Subsystem->RegisterCurveComponent(this);
            RefreshFromSubsystem();
        }
    }
}

void ULiveAudioEmotionCurveComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    if (Subsystem.IsValid())
    {
        Subsystem->UnregisterCurveComponent(this);
    }
    Super::EndPlay(EndPlayReason);
}

void ULiveAudioEmotionCurveComponent::RefreshFromSubsystem()
{
    if (!Subsystem.IsValid())
    {
        return;
    }
    const TMap<FName, float>& Curves = Subsystem->GetCurveValues();
    for (const TPair<FName, float>& Pair : Curves)
    {
        SetCurveValue(Pair.Key, Pair.Value);
    }
}
