# Create __init__.py files for all Python packages and .gitkeep for empty dirs

# Python package init files
$initFiles = @(
    'backend\app\__init__.py',
    'backend\app\api\__init__.py',
    'backend\app\api\routes\__init__.py',
    'backend\app\agent\__init__.py',
    'backend\app\models\__init__.py',
    'backend\app\preprocessing\__init__.py',
    'backend\app\geospatial\__init__.py',
    'backend\app\evaluation\__init__.py',
    'backend\app\reports\__init__.py',
    'backend\app\database\__init__.py',
    'backend\app\schemas\__init__.py',
    'backend\app\services\__init__.py',
    'backend\app\utils\__init__.py',
    'backend\tests\__init__.py',
    'backend\tests\unit\__init__.py',
    'backend\tests\integration\__init__.py',
    'backend\tests\agent\__init__.py',
    'backend\tests\models\__init__.py',
    'backend\tests\preprocessing\__init__.py',
    'backend\tests\api\__init__.py'
)

foreach ($f in $initFiles) {
    New-Item -ItemType File -Path $f -Force | Out-Null
    Set-Content -Path $f -Value '"""SatQuery AI package."""'
    Write-Host "Created: $f"
}

# .gitkeep for empty directories
$gitkeepDirs = @(
    'models\vqa',
    'models\captioning',
    'models\grounding',
    'models\change_detection',
    'models\change_vqa',
    'models\optical_sar',
    'datasets\bigearthnet_txt',
    'datasets\vrsbench',
    'datasets\rsvqa',
    'datasets\cdvqa',
    'outputs\processed',
    'outputs\evidence',
    'outputs\evaluation',
    'reports',
    'notebooks',
    'docker',
    'uploads',
    'backend\app\reports\templates'
)

foreach ($dir in $gitkeepDirs) {
    $gitkeep = Join-Path $dir '.gitkeep'
    New-Item -ItemType File -Path $gitkeep -Force | Out-Null
    Write-Host "Created: $gitkeep"
}

Write-Host "`nAll placeholder files created."
