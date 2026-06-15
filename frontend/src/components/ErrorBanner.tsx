interface ErrorBannerProps {
  message: string;
}

export function ErrorBanner({ message }: ErrorBannerProps) {
  return (
    <div className="rounded-md border border-red-800 bg-red-950/40 px-4 py-3 text-sm text-red-300">
      <span className="font-semibold">Error: </span>
      {message}
    </div>
  );
}
