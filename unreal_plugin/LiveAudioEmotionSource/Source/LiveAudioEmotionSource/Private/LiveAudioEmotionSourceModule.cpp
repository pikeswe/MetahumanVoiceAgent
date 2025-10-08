#include "Modules/ModuleManager.h"

class FLiveAudioEmotionSourceModule : public IModuleInterface
{
public:
    virtual void StartupModule() override {}
    virtual void ShutdownModule() override {}
};

IMPLEMENT_MODULE(FLiveAudioEmotionSourceModule, LiveAudioEmotionSource)
