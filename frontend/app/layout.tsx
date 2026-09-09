import type { Metadata } from "next";
import Link from "next/link";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

import { PlayerBar } from "@/components/PlayerBar";
import { PlayerProvider } from "@/components/PlayerProvider";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Music Discovery Engine",
  description:
    "Search Creative-Commons music by how it sounds, not by how many times it has been played.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-neutral-950 text-white">
        <PlayerProvider>
          <header className="border-b border-white/10">
            <nav className="mx-auto flex max-w-5xl items-center gap-6 px-4 py-4">
              <Link href="/" className="mr-2 text-sm font-semibold tracking-tight">
                ◍ Music Discovery Engine
              </Link>
              <Link href="/browse" className="text-xs text-white/50 transition hover:text-white">
                Browse
              </Link>
              <Link
                href="/experiments"
                className="text-xs text-white/50 transition hover:text-white"
              >
                Experiments
              </Link>
            </nav>
          </header>

          {/* Bottom padding keeps the last row clear of the fixed player bar. */}
          <main className="mx-auto w-full max-w-5xl flex-1 px-4 py-8 pb-32">
            {children}
          </main>

          <PlayerBar />
        </PlayerProvider>
      </body>
    </html>
  );
}
