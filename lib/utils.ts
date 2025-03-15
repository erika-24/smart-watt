import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatPower(value: number, unit = "kW"): string {
  return `${value.toFixed(1)} ${unit}`
}

export function formatPercentage(value: number): string {
  return `${value.toFixed(0)}%`
}

export function formatCurrency(value: number, currency = "$"): string {
  return `${currency}${value.toFixed(2)}`
}

export function formatDate(date: Date): string {
  return date.toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  })
}

export function formatTime(date: Date): string {
  return date.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
  })
}

export function formatDateTime(date: Date): string {
  return `${formatDate(date)} ${formatTime(date)}`
}

