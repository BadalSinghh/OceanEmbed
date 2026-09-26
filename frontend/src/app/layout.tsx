import type { Metadata } from "next";
import "./globals.css";
import { Nav } from "@/components/Nav";

export const metadata: Metadata = {
  title: {
    default: "OceanEmbed — Subsurface Ocean Temperature Reconstruction",
    template: "%s | OceanEmbed",
  },
  description:
    "Satellite embedding-based deep learning framework for reconstruction of subsurface ocean temperature from surface satellite observations over the Bay of Bengal.",
  keywords: [
    "ocean",
    "deep learning",
    "satellite",
    "subsurface temperature",
    "Bay of Bengal",
    "FNO",
    "CBAM",
  ],
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <Nav />
        <main>{children}</main>
      </body>
    </html>
  );
}
