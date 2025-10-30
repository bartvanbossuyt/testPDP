# 🎯 Commit Message Linting Setup Package

This folder contains everything you need to set up automatic commit message linting in any Git repository.

---

## 📦 What's Included

```
commit-linting-setup/
├── .commitlintrc.js           # Commit message rules
├── .husky/                    # Git hooks folder
│   ├── commit-msg             # The validation hook
│   └── _/                     # Husky helper files
├── package.json               # Node.js dependencies
├── COMMIT_GUIDELINES.md       # Complete reference guide
├── COMMIT_TUTORIAL.md         # Step-by-step tutorial
└── SETUP_INSTRUCTIONS.md      # This file
```

---

## 🚀 How to Install in a New Repository

### Step 1: Copy Files

Copy all files from this folder to your target repository's root directory:

```powershell
# Navigate to your target repository
cd path\to\your\repo

# Copy all files (adjust the path to this folder)
Copy-Item "path\to\commit-linting-setup\*" . -Recurse -Force
```

### Step 2: Install Dependencies

Make sure Node.js is installed, then run:

```powershell
npm install
```

This will install:
- `@commitlint/cli`
- `@commitlint/config-conventional`
- `husky`

### Step 3: Verify Setup

Test if it works:

```powershell
# Try an invalid commit (should be rejected)
git commit -m "test" --allow-empty

# Try a valid commit (should be accepted)
git commit -m "test: verify commit linting setup" --allow-empty
```

---

## 🔧 Alternative: Manual Step-by-Step Setup

If you prefer to set it up manually:

### 1. Initialize npm (if not already done)
```powershell
npm init -y
```

### 2. Install packages
```powershell
npm install --save-dev @commitlint/cli @commitlint/config-conventional husky
```

### 3. Copy configuration files
- Copy `.commitlintrc.js` to root
- Copy `.husky/` folder to root

### 4. Update package.json
Add this script to your `package.json`:
```json
"scripts": {
  "prepare": "husky install"
}
```

### 5. Run prepare script
```powershell
npm run prepare
```

---

## 📝 Customizing for Your Project

### Modify Commit Types

Edit `.commitlintrc.js` and change the `type-enum` array:

```javascript
'type-enum': [
  2,
  'always',
  [
    'feat',      // Add or remove types as needed
    'fix',
    'docs',
    // ... more types
  ]
]
```

### Adjust Message Length

In `.commitlintrc.js`:

```javascript
'header-max-length': [2, 'always', 100],  // Change 100 to your preference
```

### Add Project-Specific Scopes

In `.commitlintrc.js`, add:

```javascript
'scope-enum': [
  2,
  'always',
  ['api', 'ui', 'database', 'auth']  // Your scopes
]
```

---

## 🗑️ How to Uninstall

If you want to remove commit linting:

```powershell
# Remove node packages
npm uninstall @commitlint/cli @commitlint/config-conventional husky

# Delete configuration files
Remove-Item .commitlintrc.js
Remove-Item .husky -Recurse
Remove-Item node_modules -Recurse
Remove-Item package.json
Remove-Item package-lock.json
```

---

## 📚 Documentation

- **COMMIT_GUIDELINES.md** - Complete reference for commit message formats
- **COMMIT_TUTORIAL.md** - Interactive tutorial with examples

---

## ⚠️ Important Notes

### Node.js Required
This setup requires Node.js and npm to be installed. Download from: https://nodejs.org/

### Add to .gitignore
Make sure your `.gitignore` includes:
```
node_modules/
package-lock.json  # Optional, some prefer to commit this
```

### PowerShell Execution Policy
If you get errors about running scripts, you may need to adjust PowerShell execution policy or use the `.cmd` versions of commands:
```powershell
npm.cmd install
npx.cmd commitlint
```

### Existing package.json
If your repository already has a `package.json`, you'll need to manually merge the dependencies and scripts from the provided file.

---

## 🆘 Troubleshooting

### Hook not running?
```powershell
# Ensure husky is installed
npm run prepare

# Check if hook file exists
ls .husky\commit-msg
```

### Commitlint not found?
```powershell
# Reinstall dependencies
npm install
```

### Still having issues?
Check that:
1. Node.js is installed: `node --version`
2. Dependencies are installed: `ls node_modules`
3. Hook is executable: Check `.husky/commit-msg` exists
4. You're in a Git repository: `git status`

---

## 🎓 Learning Resources

1. [Conventional Commits](https://www.conventionalcommits.org/)
2. [Commitlint Documentation](https://commitlint.js.org/)
3. [Husky Documentation](https://typicode.github.io/husky/)

---

## ✅ Quick Checklist

After installation, verify:
- [ ] `node_modules/` folder exists
- [ ] `.husky/commit-msg` file exists
- [ ] `.commitlintrc.js` file exists
- [ ] `package.json` has commitlint and husky dependencies
- [ ] Invalid commit message gets rejected
- [ ] Valid commit message gets accepted

---

**Happy committing! 🎉**