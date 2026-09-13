# Create all directories for SatQuery AI

# Backend
$backendDirs = @(
    'backend\app\api\routes',
    'backend\app\agent',
    'backend\app\models',
    'backend\app\preprocessing',
    'backend\app\geospatial',
    'backend\app\evaluation',
    'backend\app\reports\templates',
    'backend\app\database',
    'backend\app\schemas',
    'backend\app\services',
    'backend\app\utils'
)

# Tests
$testDirs = @(
    'backend\tests\unit',
    'backend\tests\integration',
    'backend\tests\agent',
    'backend\tests\models',
    'backend\tests\preprocessing',
    'backend\tests\api'
)

# Frontend
$frontendDirs = @(
    'frontend\src\components',
    'frontend\src\pages',
    'frontend\src\services',
    'frontend\src\types',
    'frontend\public'
)

# Other
$otherDirs = @(
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
    'scripts',
    'docker',
    'uploads'
)

$allDirs = $backendDirs + $testDirs + $frontendDirs + $otherDirs

foreach ($dir in $allDirs) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
    Write-Host "Created: $dir"
}

Write-Host "`nAll directories created successfully."
