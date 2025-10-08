using UnrealBuildTool;

public class LiveAudioEmotionSource : ModuleRules
{
    public LiveAudioEmotionSource(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new[]
        {
            "Core",
            "CoreUObject",
            "Engine",
            "InputCore",
            "AudioMixer",
            "WebSockets",
            "HTTP",
            "Json",
            "JsonUtilities",
        });

        PrivateDependencyModuleNames.AddRange(new[]
        {
            "Projects",
            "Slate",
            "SlateCore"
        });
    }
}
