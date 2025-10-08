#pragma once

#include "CoreMinimal.h"
#include "Animation/CurveSourceComponent.h"
#include "LiveAudioEmotionCurveComponent.generated.h"

UCLASS(ClassGroup = (MetaHuman), Blueprintable, meta = (BlueprintSpawnableComponent))
class ULiveAudioEmotionCurveComponent : public UCurveSourceComponent
{
    GENERATED_BODY()

public:
    ULiveAudioEmotionCurveComponent();

    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

    void RefreshFromSubsystem();

private:
    TWeakObjectPtr<class ULiveAudioEmotionSourceSubsystem> Subsystem;
};
