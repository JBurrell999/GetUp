import { CaptureProvider } from "@/components/CaptureProvider";
import { Footer, Hero, Setup } from "@/components/Intro";
import { JudgeDialog } from "@/components/JudgeDialog";
import { ShotList } from "@/components/ShotList";
import { TaskSheet } from "@/components/TaskSheet";
import { TopBar } from "@/components/TopBar";
import { publicConfig } from "@/lib/config";

// Read the Blob/event-code settings per request, so changing them never leaves a stale page.
export const dynamic = "force-dynamic";

export default function Page() {
  return (
    <CaptureProvider config={publicConfig()}>
      <TopBar />
      <main>
        <Hero />
        <Setup />
        <ShotList />
      </main>
      <Footer />
      <TaskSheet />
      <JudgeDialog />
    </CaptureProvider>
  );
}
