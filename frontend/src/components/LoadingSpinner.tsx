export function LoadingSpinner() {
  return (
    <div className="flex items-center justify-center py-24">
      <div className="h-8 w-8 animate-spin rounded-full border-2 border-slate-600 border-t-blue-500" />
    </div>
  );
}
