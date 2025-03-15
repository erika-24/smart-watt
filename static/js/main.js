// Main JavaScript file for SmartWatt

// Document ready
$(document).ready(() => {
  // Toggle sidebar
  $("#sidebarCollapse").on("click", () => {
    $("#sidebar").toggleClass("collapsed")
  })

  // Initialize tooltips
  var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
  var tooltipList = tooltipTriggerList.map((tooltipTriggerEl) => new bootstrap.Tooltip(tooltipTriggerEl))

  // Initialize popovers
  var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'))
  var popoverList = popoverTriggerList.map((popoverTriggerEl) => new bootstrap.Popover(popoverTriggerEl))

  // Auto-hide alerts after 5 seconds
  setTimeout(() => {
    $(".alert-auto-dismiss").fadeOut("slow")
  }, 5000)
})

// Format number with commas
function formatNumber(num) {
  return num.toString().replace(/(\d)(?=(\d{3})+(?!\d))/g, "$1,")
}

// Format date
function formatDate(date) {
  const options = { year: "numeric", month: "short", day: "numeric" }
  return new Date(date).toLocaleDateString(undefined, options)
}

// Format time
function formatTime(date) {
  const options = { hour: "2-digit", minute: "2-digit" }
  return new Date(date).toLocaleTimeString(undefined, options)
}

// Format power (kW)
function formatPower(power) {
  return power.toFixed(1) + " kW"
}

// Format energy (kWh)
function formatEnergy(energy) {
  return energy.toFixed(1) + " kWh"
}

// Format percentage
function formatPercentage(percentage) {
  return percentage.toFixed(1) + "%"
}

// Format currency
function formatCurrency(amount) {
  return "$" + amount.toFixed(2)
}

// Show loading spinner
function showLoading(elementId, message = "Loading...") {
  const element = document.getElementById(elementId)
  if (element) {
    element.innerHTML = `
            <div class="d-flex justify-content-center align-items-center p-5">
                <div class="spinner-border text-primary me-3" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <span>${message}</span>
            </div>
        `
  }
}

// Show error message
function showError(elementId, message = "An error occurred.") {
  const element = document.getElementById(elementId)
  if (element) {
    element.innerHTML = `
            <div class="alert alert-danger" role="alert">
                <i class="fas fa-exclamation-circle me-2"></i>
                ${message}
            </div>
        `
  }
}

// Copy to clipboard
function copyToClipboard(text) {
  const textarea = document.createElement("textarea")
  textarea.value = text
  document.body.appendChild(textarea)
  textarea.select()
  document.execCommand("copy")
  document.body.removeChild(textarea)

  // Show toast notification
  const toast = new bootstrap.Toast(document.getElementById("copyToast"))
  toast.show()
}

// Download data as JSON
function downloadJson(data, filename) {
  const jsonStr = JSON.stringify(data, null, 2)
  const blob = new Blob([jsonStr], { type: "application/json" })
  const url = URL.createObjectURL(blob)

  const a = document.createElement("a")
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()

  setTimeout(() => {
    document.body.removeChild(a)
    window.URL.revokeObjectURL(url)
  }, 0)
}

// Download data as CSV
function downloadCsv(data, filename) {
  // Convert data to CSV format
  let csvContent = ""

  // Add headers
  if (data.length > 0) {
    csvContent += Object.keys(data[0]).join(",") + "\n"
  }

  // Add rows
  data.forEach((row) => {
    const values = Object.values(row).map((value) => {
      // Handle strings with commas
      if (typeof value === "string" && value.includes(",")) {
        return `"${value}"`
      }
      return value
    })
    csvContent += values.join(",") + "\n"
  })

  // Create download link
  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" })
  const url = URL.createObjectURL(blob)

  const a = document.createElement("a")
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()

  setTimeout(() => {
    document.body.removeChild(a)
    window.URL.revokeObjectURL(url)
  }, 0)
}

