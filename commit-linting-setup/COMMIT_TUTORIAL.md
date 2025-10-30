# 🎓 Commit Guidelines Tutorial

This tutorial will walk you through making your first commits with the new conventional commit message system.

---

## 📚 Quick Start Guide

### Step 1: Understanding the Basic Format

Every commit message must follow this pattern:
```
<type>: <description>
```

**Example:**
```bash
feat: add new clustering algorithm
fix: correct matrix calculation error
docs: update installation instructions
```

---

## 🎯 Step-by-Step: Making Your First Commit

### Example 1: Adding a New Feature

Let's say you added a new visualization function to `N_VA_HeatMap.py`.

**1. Stage your changes:**
```powershell
git add scripts/N_VA_HeatMap.py
```

**2. Commit with proper message:**
```powershell
git commit -m "feat(vis): add color gradient option to heatmap"
```

✅ **Result:** Commit accepted!

---

### Example 2: Fixing a Bug

You fixed a calculation error in `N_PDP.py`.

**1. Stage the fix:**
```powershell
git add scripts/N_PDP.py
```

**2. Commit:**
```powershell
git commit -m "fix(pdp): correct distance matrix calculation for edge cases"
```

✅ **Result:** Commit accepted!

---

### Example 3: Updating Documentation

You updated the README with better examples.

**1. Stage the changes:**
```powershell
git add README.md
```

**2. Commit:**
```powershell
git commit -m "docs: add usage examples to README"
```

✅ **Result:** Commit accepted!

---

## ❌ Common Mistakes and How to Fix Them

### Mistake 1: Missing Type
```powershell
git commit -m "added new feature"
```
**Error:** `type may not be empty`

**Fix:**
```powershell
git commit -m "feat: add new feature"
```

---

### Mistake 2: Wrong Capitalization
```powershell
git commit -m "FEAT: add new feature"
```
**Error:** `type must be lowercase`

**Fix:**
```powershell
git commit -m "feat: add new feature"
```

---

### Mistake 3: Period at the End
```powershell
git commit -m "feat: add new feature."
```
**Error:** `subject may not end with period`

**Fix:**
```powershell
git commit -m "feat: add new feature"
```

---

### Mistake 4: Too Vague
```powershell
git commit -m "fix: fixed stuff"
```
**Technically valid, but NOT helpful!**

**Better:**
```powershell
git commit -m "fix(scripts): resolve import error in N_Moving_Objects"
```

---

## 🎨 Real-World Examples for Your Project

### Adding New Scripts
```powershell
git commit -m "feat(scripts): add N_VA_TSNE visualization module"
git commit -m "feat(data): create new dataset preprocessing script"
```

### Fixing Bugs
```powershell
git commit -m "fix(gui): resolve file path issue in GUI selector"
git commit -m "fix(pdp): correct buffer calculation in rough mode"
git commit -m "fix(vis): fix heatmap color scaling"
```

### Improving Performance
```powershell
git commit -m "perf(pdp): optimize distance matrix computation"
git commit -m "perf(scripts): reduce memory usage in large datasets"
```

### Documentation Updates
```powershell
git commit -m "docs: add commit guidelines tutorial"
git commit -m "docs(readme): update installation steps"
git commit -m "docs(scripts): add docstrings to utility functions"
```

### Code Refactoring
```powershell
git commit -m "refactor(av): reorganize configuration variables"
git commit -m "refactor(scripts): extract common functions to utils module"
```

### Adding Tests
```powershell
git commit -m "test(pdp): add unit tests for distance calculations"
git commit -m "test(vis): add integration tests for visualizations"
```

### Configuration/Dependencies
```powershell
git commit -m "chore: update requirements.txt with missing packages"
git commit -m "chore: add commit linting with husky and commitlint"
git commit -m "build: configure husky for git hooks"
```

### Style/Formatting
```powershell
git commit -m "style(scripts): format code with black"
git commit -m "style: fix indentation in multiple files"
```

---

## 📝 Multi-line Commits (Advanced)

For more complex changes, you can add a detailed body:

```powershell
git commit -m "feat(pdp): add support for dynamic buffer sizing

This change allows users to specify different buffer sizes
for different configurations. The buffer size can now be
set per-configuration in the settings file.

Closes #42"
```

**Format:**
- Line 1: `<type>(<scope>): <subject>` (max 100 chars)
- Line 2: Blank line
- Line 3+: Detailed explanation
- Footer: Reference issues (optional)

---

## 🔄 If You Make a Mistake

### Already committed with wrong message?

**Option 1: Fix the last commit (if not pushed yet)**
```powershell
git commit --amend -m "feat: correct commit message"
```

**Option 2: Already pushed?**
- Don't worry! Just make sure your next commit follows the rules
- The hook only validates NEW commits

---

## 🛠️ Practical Workflow

### Daily Work Pattern:

1. **Make your code changes**
2. **Review what changed:**
   ```powershell
   git status
   git diff
   ```

3. **Stage your changes:**
   ```powershell
   git add <files>
   # or add everything:
   git add .
   ```

4. **Think about the commit type:**
   - Did I add something new? → `feat`
   - Did I fix a bug? → `fix`
   - Did I update docs? → `docs`
   - Did I refactor? → `refactor`

5. **Commit with proper message:**
   ```powershell
   git commit -m "<type>(<scope>): <description>"
   ```

6. **If rejected, read the error and try again**

7. **Push your changes:**
   ```powershell
   git push
   ```

---

## 🎓 Practice Exercise

Try committing these scenarios:

### Scenario 1
You updated the `requirements.txt` file to add missing packages.

<details>
<summary>Click to see answer</summary>

```powershell
git commit -m "chore: update requirements.txt with missing dependencies"
```
</details>

### Scenario 2
You fixed a bug where the GUI wasn't loading files correctly.

<details>
<summary>Click to see answer</summary>

```powershell
git commit -m "fix(gui): resolve file loading error in GUI interface"
```
</details>

### Scenario 3
You added a new t-SNE visualization script.

<details>
<summary>Click to see answer</summary>

```powershell
git commit -m "feat(vis): add t-SNE dimensionality reduction visualization"
```
</details>

---

## 💡 Pro Tips

1. **Commit often:** Make small, focused commits rather than large ones
2. **Be specific:** "fix login bug" → "fix(auth): resolve null pointer in login validation"
3. **Use scopes:** They help categorize changes (scripts, gui, data, vis, docs)
4. **Present tense:** Use "add" not "added", "fix" not "fixed"
5. **No periods:** Don't end the subject with a period
6. **Keep it short:** Aim for 50-72 characters in the subject

---

## 🆘 Getting Help

### Check if your message would be valid:
```powershell
echo "feat: my commit message" | npx.cmd commitlint
```

### See the last few commits as examples:
```powershell
git log --oneline -10
```

### View detailed commit history:
```powershell
git log
```

---

## 📖 Quick Reference Card

```
Types:
  feat     → New feature
  fix      → Bug fix
  docs     → Documentation
  style    → Formatting
  refactor → Code restructuring
  test     → Tests
  chore    → Maintenance
  perf     → Performance
  ci       → CI/CD
  build    → Build system
  revert   → Undo changes

Format:
  <type>(<scope>): <subject>
  
Rules:
  ✅ Lowercase type
  ✅ No period at end
  ✅ Max 100 characters
  ✅ Present tense
  ✅ Clear and specific
  
  ❌ No type
  ❌ Uppercase type
  ❌ Period at end
  ❌ Too vague
```

---

## 🎉 You're Ready!

Start making commits following these guidelines. The system will help you by rejecting invalid messages with clear error messages. Don't be discouraged if you get rejections at first - it's a learning process!

**Remember:** The goal is to make your commit history clear, searchable, and professional. Future you (and your collaborators) will thank you! 🙏