#pragma once

#include "CoreMinimal.h"
#include "Subsystems/GameInstanceSubsystem.h"
#include "LiveAudioEmotionSourceSubsystem.generated.h"

class USoundWaveProcedural;
class ULiveAudioEmotionCurveComponent;
class IWebSocket;

UCLASS()
class ULiveAudioEmotionSourceSubsystem : public UGameInstanceSubsystem
{
    GENERATED_BODY()

public:
    ULiveAudioEmotionSourceSubsystem();

    virtual void Deinitialize() override;

    bool Connect(const FString& Url);
    void Disconnect();

    void SendTextRequest(const FString& Url, const FString& Prompt);

    UFUNCTION(BlueprintCallable, Category = "LiveAudioEmotion")
    USoundWaveProcedural* GetProceduralSoundWave() const { return ProceduralSoundWave; }

    void RegisterCurveComponent(ULiveAudioEmotionCurveComponent* Component);
    void UnregisterCurveComponent(ULiveAudioEmotionCurveComponent* Component);

    const TMap<FName, float>& GetCurveValues() const { return CurveValues; }

private:
    void HandleTextMessage(const FString& Message);
    void HandleBinaryMessage(const void* Data, SIZE_T Size, SIZE_T BytesRemaining);

    void ConfigureWave(int32 InSampleRate, int32 InChannels, int32 InChunkMs);
    void UpdateCurves(const TMap<FName, float>& InCurves);

    TSharedPtr<IWebSocket> Socket;

    UPROPERTY()
    USoundWaveProcedural* ProceduralSoundWave;

    TWeakObjectPtr<ULiveAudioEmotionCurveComponent> RegisteredCurveComponent;

    TMap<FName, float> CurveValues;
    int32 SampleRate;
    int32 Channels;
    int32 ChunkMs;
};
