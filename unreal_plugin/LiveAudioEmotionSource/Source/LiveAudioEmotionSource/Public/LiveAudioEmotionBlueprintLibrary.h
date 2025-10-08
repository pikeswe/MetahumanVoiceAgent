#pragma once

#include "Kismet/BlueprintFunctionLibrary.h"
#include "LiveAudioEmotionBlueprintLibrary.generated.h"

UCLASS()
class ULiveAudioEmotionBlueprintLibrary : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()

public:
    UFUNCTION(BlueprintCallable, Category = "LiveAudioEmotion", meta = (WorldContext = "WorldContextObject"))
    static bool ConnectVoiceServer(UObject* WorldContextObject, const FString& Url);

    UFUNCTION(BlueprintCallable, Category = "LiveAudioEmotion", meta = (WorldContext = "WorldContextObject"))
    static void DisconnectVoiceServer(UObject* WorldContextObject);

    UFUNCTION(BlueprintCallable, Category = "LiveAudioEmotion", meta = (WorldContext = "WorldContextObject"))
    static void SendTextToAgent(UObject* WorldContextObject, const FString& Text, const FString& AgentUrl = TEXT("http://127.0.0.1:17860/ask"));

    UFUNCTION(BlueprintCallable, Category = "LiveAudioEmotion", meta = (WorldContext = "WorldContextObject"))
    static USoundWaveProcedural* GetProceduralSoundWave(UObject* WorldContextObject);
};
