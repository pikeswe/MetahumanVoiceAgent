#include "LiveAudioEmotionSourceSubsystem.h"

#include "Async/Async.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "HttpModule.h"
#include "IWebSocket.h"
#include "Interfaces/IHttpRequest.h"
#include "Interfaces/IHttpResponse.h"
#include "Sound/SoundWaveProcedural.h"
#include "WebSocketsModule.h"
#include "LiveAudioEmotionCurveComponent.h"

namespace
{
static const TCHAR* DefaultCurveNames[] = {
    TEXT("Emotion_Neutral"),
    TEXT("Emotion_Happy"),
    TEXT("Emotion_Sad"),
    TEXT("Emotion_Angry"),
    TEXT("Emotion_Surprised"),
    TEXT("Prosody_Rate"),
    TEXT("Prosody_Intensity")
};
}

ULiveAudioEmotionSourceSubsystem::ULiveAudioEmotionSourceSubsystem()
    : ProceduralSoundWave(nullptr)
    , SampleRate(22050)
    , Channels(1)
    , ChunkMs(20)
{
    CurveValues.Reserve(7);
    for (const TCHAR* Name : DefaultCurveNames)
    {
        CurveValues.Add(FName(Name), 0.f);
    }
    CurveValues.FindOrAdd(TEXT("Emotion_Neutral")) = 1.f;
}

void ULiveAudioEmotionSourceSubsystem::Deinitialize()
{
    Disconnect();
    Super::Deinitialize();
}

bool ULiveAudioEmotionSourceSubsystem::Connect(const FString& Url)
{
    if (Socket.IsValid())
    {
        Socket->Close();
        Socket.Reset();
    }

    FModuleManager::LoadModuleChecked<FWebSocketsModule>(TEXT("WebSockets"));
    TSharedPtr<IWebSocket> NewSocket = FWebSocketsModule::Get().CreateWebSocket(Url);
    if (!NewSocket.IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to create WebSocket for %s"), *Url);
        return false;
    }

    NewSocket->OnConnected().AddLambda([]()
    {
        UE_LOG(LogTemp, Log, TEXT("Connected to voice server."));
    });

    NewSocket->OnMessage().AddUObject(this, &ULiveAudioEmotionSourceSubsystem::HandleTextMessage);
    NewSocket->OnRawMessage().AddUObject(this, &ULiveAudioEmotionSourceSubsystem::HandleBinaryMessage);
    NewSocket->OnConnectionError().AddLambda([](const FString& Error)
    {
        UE_LOG(LogTemp, Error, TEXT("Voice socket error: %s"), *Error);
    });
    NewSocket->OnClosed().AddLambda([this](int32, const FString&, bool)
    {
        AsyncTask(ENamedThreads::GameThread, [this]()
        {
            if (ProceduralSoundWave)
            {
                ProceduralSoundWave->ResetAudio();
            }
        });
    });

    Socket = NewSocket;
    Socket->Connect();
    return true;
}

void ULiveAudioEmotionSourceSubsystem::Disconnect()
{
    if (Socket.IsValid())
    {
        Socket->Close();
        Socket.Reset();
    }
    if (ProceduralSoundWave)
    {
        ProceduralSoundWave->ResetAudio();
    }
}

void ULiveAudioEmotionSourceSubsystem::SendTextRequest(const FString& Url, const FString& Prompt)
{
    if (Prompt.IsEmpty())
    {
        return;
    }

    FHttpModule& Http = FHttpModule::Get();
    TSharedRef<IHttpRequest, ESPMode::ThreadSafe> Request = Http.CreateRequest();
    Request->SetURL(Url);
    Request->SetVerb(TEXT("POST"));
    Request->SetHeader(TEXT("Content-Type"), TEXT("application/json"));
    TSharedPtr<FJsonObject> PayloadObj = MakeShared<FJsonObject>();
    PayloadObj->SetStringField(TEXT("prompt"), Prompt);
    FString Payload;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Payload);
    FJsonSerializer::Serialize(PayloadObj.ToSharedRef(), Writer);
    Request->SetContentAsString(Payload);
    Request->OnProcessRequestComplete().BindLambda([](FHttpRequestPtr Req, FHttpResponsePtr Response, bool bWasSuccessful)
    {
        if (!bWasSuccessful || !Response.IsValid() || !EHttpResponseCodes::IsOk(Response->GetResponseCode()))
        {
            UE_LOG(LogTemp, Warning, TEXT("SendTextToAgent failed: %s"), Response.IsValid() ? *Response->GetContentAsString() : TEXT("No Response"));
        }
    });
    Request->ProcessRequest();
}

void ULiveAudioEmotionSourceSubsystem::RegisterCurveComponent(ULiveAudioEmotionCurveComponent* Component)
{
    RegisteredCurveComponent = Component;
}

void ULiveAudioEmotionSourceSubsystem::UnregisterCurveComponent(ULiveAudioEmotionCurveComponent* Component)
{
    if (RegisteredCurveComponent.Get() == Component)
    {
        RegisteredCurveComponent.Reset();
    }
}

void ULiveAudioEmotionSourceSubsystem::ConfigureWave(int32 InSampleRate, int32 InChannels, int32 InChunkMs)
{
    SampleRate = InSampleRate;
    Channels = InChannels;
    ChunkMs = InChunkMs;
    if (!ProceduralSoundWave)
    {
        ProceduralSoundWave = NewObject<USoundWaveProcedural>(this);
    }
    ProceduralSoundWave->SampleRate = SampleRate;
    ProceduralSoundWave->NumChannels = Channels;
    ProceduralSoundWave->Duration = INDEFINITELY_LOOPING_DURATION;
    ProceduralSoundWave->SoundGroup = SOUNDGROUP_Voice;
    ProceduralSoundWave->bLooping = false;
    ProceduralSoundWave->SetSampleRate(SampleRate);
    ProceduralSoundWave->FlushAudio();
}

void ULiveAudioEmotionSourceSubsystem::UpdateCurves(const TMap<FName, float>& InCurves)
{
    CurveValues = InCurves;
    if (ULiveAudioEmotionCurveComponent* CurveComponent = RegisteredCurveComponent.Get())
    {
        CurveComponent->RefreshFromSubsystem();
    }
}

void ULiveAudioEmotionSourceSubsystem::HandleTextMessage(const FString& Message)
{
    TSharedPtr<FJsonObject> JsonObject;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Message);
    if (!FJsonSerializer::Deserialize(Reader, JsonObject) || !JsonObject.IsValid())
    {
        UE_LOG(LogTemp, Warning, TEXT("Invalid JSON from voice server: %s"), *Message);
        return;
    }

    FString Type = JsonObject->GetStringField(TEXT("type"));
    if (Type == TEXT("start"))
    {
        int32 InSampleRate = JsonObject->GetIntegerField(TEXT("sample_rate"));
        int32 InChannels = JsonObject->GetIntegerField(TEXT("channels"));
        int32 InChunk = JsonObject->GetIntegerField(TEXT("chunk_ms"));
        AsyncTask(ENamedThreads::GameThread, [this, InSampleRate, InChannels, InChunk]()
        {
            ConfigureWave(InSampleRate, InChannels, InChunk);
        });
    }
    else if (Type == TEXT("emotion"))
    {
        TMap<FName, float> Incoming;
        for (const TPair<FString, TSharedPtr<FJsonValue>>& Pair : JsonObject->Values)
        {
            if (Pair.Key == TEXT("type"))
            {
                continue;
            }
            Incoming.Add(FName(*Pair.Key), static_cast<float>(Pair.Value->AsNumber()));
        }
        AsyncTask(ENamedThreads::GameThread, [this, Incoming]()
        {
            UpdateCurves(Incoming);
        });
    }
    else if (Type == TEXT("end"))
    {
        UE_LOG(LogTemp, Log, TEXT("Voice stream ended."));
    }
}

void ULiveAudioEmotionSourceSubsystem::HandleBinaryMessage(const void* Data, SIZE_T Size, SIZE_T BytesRemaining)
{
    TArray<uint8> AudioData;
    AudioData.AddUninitialized(Size);
    FMemory::Memcpy(AudioData.GetData(), Data, Size);
    AsyncTask(ENamedThreads::GameThread, [this, AudioData]() mutable
    {
        if (ProceduralSoundWave)
        {
            ProceduralSoundWave->QueueAudio(AudioData.GetData(), AudioData.Num());
        }
    });
}
