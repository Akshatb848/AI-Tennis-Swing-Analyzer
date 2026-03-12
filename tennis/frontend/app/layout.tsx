import type { Metadata, Viewport } from 'next';
import { Inter } from 'next/font/google';
import '@/styles/globals.css';
import { Providers } from './providers';
import { Navbar } from '@/components/Navbar';

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' });

export const metadata: Metadata = {
    title: {
        default: 'TennisIQ — Match Intelligence Platform',
        template: '%s | TennisIQ',
    },
    description:
        'Upload your tennis match video and get automatic shot detection, rally analysis, and player statistics.',
    keywords: ['tennis analytics', 'match analysis', 'shot detection', 'player tracking'],
};

export const viewport: Viewport = {
    themeColor: '#0a0a0f',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
    return (
        <html lang="en" className={inter.variable}>
            <body className="min-h-screen flex flex-col">
                <Providers>
                    <Navbar />
                    <main className="flex-1 pt-16">{children}</main>
                    <footer className="border-t border-white/[0.06] py-8 text-center text-xs text-white/30">
                        © {new Date().getFullYear()} TennisIQ. Match intelligence for every level.
                    </footer>
                </Providers>
            </body>
        </html>
    );
}
