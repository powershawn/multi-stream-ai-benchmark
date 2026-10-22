param([switch]$AsBase64)

$ErrorActionPreference = 'Stop'
$snapshotWarnings = @()
$cpuInfo = $null
$visibleRam = $null
$installedRam = $null
$osInfo = $null
try {
    $processors = @(Get-CimInstance Win32_Processor)
    $cpuInfo = [ordered]@{
        name = ($processors.Name -join ' / ')
        physical_cores = ($processors | Measure-Object NumberOfCores -Sum).Sum
        logical_processors = ($processors | Measure-Object NumberOfLogicalProcessors -Sum).Sum
    }
} catch { $snapshotWarnings += "CPU CIM unavailable: $($_.Exception.Message)" }
try { $visibleRam = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory }
catch { $snapshotWarnings += "Visible RAM CIM unavailable: $($_.Exception.Message)" }
try { $installedRam = (Get-CimInstance Win32_PhysicalMemory | Measure-Object Capacity -Sum).Sum }
catch { $snapshotWarnings += "Installed RAM CIM unavailable: $($_.Exception.Message)" }
try {
    $windowsOs = Get-CimInstance Win32_OperatingSystem
    $osInfo = [ordered]@{ caption = $windowsOs.Caption; version = $windowsOs.Version }
} catch { $snapshotWarnings += "OS CIM unavailable: $($_.Exception.Message)" }
$snapshot = [ordered]@{
    schema_version = 1
    source = 'Windows CIM via run.ps1'
    collected_at_utc = [DateTime]::UtcNow.ToString("yyyy-MM-dd'T'HH:mm:ss.ffffff'Z'")
    cpu = $(if ($cpuInfo) { $cpuInfo } else { @{} })
    ram = [ordered]@{ installed_bytes = $installedRam; visible_total_bytes = $visibleRam }
    os = $(if ($osInfo) { $osInfo } else { @{} })
    warnings = @($snapshotWarnings)
}
$snapshotJson = $snapshot | ConvertTo-Json -Depth 6 -Compress
if ($AsBase64) {
    [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($snapshotJson))
} else {
    $snapshotJson
}
