$ErrorActionPreference = 'Stop'

$source = 'C:\tmp\robotics_reference.docx'
if (-not $source) {
    throw 'DOCX input not found.'
}
$outputDir = 'C:\tmp\robotics_doc_render'
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
$pdfPath = Join-Path $outputDir 'robotics_reference.pdf'

$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open($source, $false, $true)
    $document.ExportAsFixedFormat($pdfPath, 17)
    Write-Output $pdfPath
}
finally {
    if ($document -ne $null) {
        $document.Close($false)
    }
    if ($word -ne $null) {
        $word.Quit()
    }
}
