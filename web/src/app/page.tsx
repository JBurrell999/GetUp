import { CaptureProvider } from "@/components/CaptureProvider";
import { Demos } from "@/components/Demos";
import { Environments } from "@/components/Environments";
import { JudgeDialog } from "@/components/JudgeDialog";
import { Mirror } from "@/components/Mirror";
import { Ideas } from "@/components/Ideas";
import { Footer, Hero, Setup } from "@/components/Sections";
import { TopBar } from "@/components/TopBar";
import { UploadSheet } from "@/components/UploadSheet";
import { publicConfig } from "@/lib/config";
import { STATIC } from "@/lib/site";
import { connection } from "next/server";

export default async function Page() {
  // Read the Blob/event-code settings per request, so changing them never leaves a stale page.
  // The GitHub Pages build is static, with no server and no uploads.
  if (!STATIC) await connection();
  return (
    <CaptureProvider config={publicConfig()}>
      <TopBar />
      <main>
        <Hero />
        <Demos />
        <Mirror />
        <Environments />
        <Setup />
        {!STATIC && <Ideas />}
      </main>
      <Footer />
      {!STATIC && (
        <>
          <UploadSheet />
          <JudgeDialog />
        </>
      )}
    </CaptureProvider>
  );
}
