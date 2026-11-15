export function formatHours(seconds = 0) {
  const hours = seconds / 3600;
  return hours > 0 && hours < 0.01 ? '<0.01' : hours.toLocaleString('en-US', { maximumFractionDigits: hours < 1 ? 2 : 1 });
}
