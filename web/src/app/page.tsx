import { CaptureProvider } from "@/components/CaptureProvider";
import { Demos } from "@/components/Demos";
import { Environments } from "@/components/Environments";
import { JudgeDialog } from "@/components/JudgeDialog";
import { Ideas } from "@/components/Ideas";
import { Footer, Hero, Setup } from "@/components/Sections";
import { TopBar } from "@/components/TopBar";
import { UploadSheet } from "@/components/UploadSheet";
import { publicConfig } from "@/lib/config";

// Read the Blob/event-code settings per request, so changing them never leaves a stale page.
export const dynamic = "force-dynamic";

export default function Page() {
  return (
    <CaptureProvider config={publicConfig()}>
      <TopBar />
      <main>
        <Hero />
        <Demos />
        <Environments />
        <Setup />
        <Ideas />
      </main>
      <Footer />
      <UploadSheet />
      <JudgeDialog />
    </CaptureProvider>
  );
}
