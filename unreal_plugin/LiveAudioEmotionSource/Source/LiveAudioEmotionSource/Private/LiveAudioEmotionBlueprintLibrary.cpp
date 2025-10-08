#include "LiveAudioEmotionBlueprintLibrary.h"

#include "LiveAudioEmotionSourceSubsystem.h"
#include "Sound/SoundWaveProcedural.h"

ULiveAudioEmotionSourceSubsystem* GetSubsystemFromContext(UObject* WorldContextObject)
{
    if (!WorldContextObject)
    {
        return nullptr;
    }
    UWorld* World = WorldContextObject->GetWorld();
    if (!World)
    {
        return nullptr;
    }
    if (UGameInstance* GameInstance = World->GetGameInstance())
    {
        return GameInstance->GetSubsystem<ULiveAudioEmotionSourceSubsystem>();
    }
    return nullptr;
}

bool ULiveAudioEmotionBlueprintLibrary::ConnectVoiceServer(UObject* WorldContextObject, const FString& Url)
{
    if (ULiveAudioEmotionSourceSubsystem* Subsystem = GetSubsystemFromContext(WorldContextObject))
    {
        return Subsystem->Connect(Url);
    }
    return false;
}

void ULiveAudioEmotionBlueprintLibrary::DisconnectVoiceServer(UObject* WorldContextObject)
{
    if (ULiveAudioEmotionSourceSubsystem* Subsystem = GetSubsystemFromContext(WorldContextObject))
    {
        Subsystem->Disconnect();
    }
}

void ULiveAudioEmotionBlueprintLibrary::SendTextToAgent(UObject* WorldContextObject, const FString& Text, const FString& AgentUrl)
{
    if (ULiveAudioEmotionSourceSubsystem* Subsystem = GetSubsystemFromContext(WorldContextObject))
    {
        Subsystem->SendTextRequest(AgentUrl, Text);
    }
}

USoundWaveProcedural* ULiveAudioEmotionBlueprintLibrary::GetProceduralSoundWave(UObject* WorldContextObject)
{
    if (ULiveAudioEmotionSourceSubsystem* Subsystem = GetSubsystemFromContext(WorldContextObject))
    {
        return Subsystem->GetProceduralSoundWave();
    }
    return nullptr;
}
