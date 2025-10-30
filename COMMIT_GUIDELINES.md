# Commit Message Guidelines

This project uses **Conventional Commits** with automated linting via Husky and Commitlint to ensure consistent and meaningful commit messages.

## 📋 Commit Message Format

Each commit message consists of a **header**, **body**, and **footer**:

```
<type>(<scope>): <subject>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```

### Required Components

#### 1. **Header** (Required)
The header has a special format that includes a **type**, **scope** (optional), and **subject**:

```
<type>(<scope>): <subject>
```

- **Maximum 100 characters**
- **Lowercase type**
- **No period at the end**

#### 2. **Body** (Optional)
- Explain **what** and **why** (not how)
- Maximum 100 characters per line
- Separate from header with blank line

#### 3. **Footer** (Optional)
- Reference issues, breaking changes
- Maximum 100 characters per line
- Separate from body with blank line

## 🏷️ Commit Types

| Type | Description | Example |
|------|-------------|---------|
| `feat` | New feature for the user | `feat: add user login functionality` |
| `fix` | Bug fix for the user | `fix: resolve dashboard loading issue` |
| `docs` | Documentation changes | `docs: update API documentation` |
| `style` | Code style (formatting, semicolons, etc.) | `style: format code with prettier` |
| `refactor` | Code refactoring (no functional changes) | `refactor: simplify authentication logic` |
| `test` | Adding or updating tests | `test: add unit tests for user service` |
| `chore` | Maintenance tasks | `chore: update dependencies` |
| `ci` | CI/CD configuration changes | `ci: add automated testing workflow` |
| `build` | Build system or external dependencies | `build: update webpack configuration` |
| `perf` | Performance improvements | `perf: optimize database queries` |
| `revert` | Reverting a previous commit | `revert: undo feat: add user login` |

## 🎯 Scope (Optional)

The scope indicates what part of the codebase is affected:

```
feat(auth): add JWT token validation
fix(ui): correct button alignment on mobile
docs(readme): add installation instructions
test(api): add integration tests for endpoints
```

Common scopes for this project might include:
- `scripts` - Changes to analysis scripts
- `gui` - GUI-related changes  
- `data` - Data processing changes
- `vis` - Visualization components
- `config` - Configuration files

## ✅ Valid Examples

### Simple commits:
```bash
feat: add new clustering algorithm
fix: correct PDP calculation error
docs: update README installation steps
style: format Python code with black
refactor: extract common utility functions
test: add tests for moving objects detection
chore: update requirements.txt
```

### With scope:
```bash
feat(scripts): implement N_VA_TSNE visualization
fix(gui): resolve file selection dialog issue
docs(api): add docstrings to all functions
refactor(data): optimize dataset creation process
test(vis): add unit tests for heatmap generation
```

### With body and footer:
```bash
feat(auth): add user authentication system

Implement JWT-based authentication with login/logout
functionality. Includes password hashing and session 
management.

Fixes #123
Closes #456
```

## ❌ Invalid Examples (Will be rejected)

```bash
# Missing type
"update documentation"

# Wrong case
"FEAT: add new feature"
"Fix: bug fix"

# Missing subject
"feat:"

# Subject ends with period
"feat: add new feature."

# Too long header (>100 characters)
"feat: this is a very long commit message that exceeds the maximum allowed length of 100 characters"

# Non-conventional format
"added some stuff"
"quick fix"
"WIP"
```

## 🚨 Breaking Changes

For breaking changes, use `!` after the type/scope:

```bash
feat!: remove deprecated API endpoints
refactor(api)!: change authentication method

BREAKING CHANGE: The old API endpoints have been removed.
Use the new v2 endpoints instead.
```

## 🔧 How It Works

1. **Husky** intercepts your commit before it's finalized
2. **Commitlint** validates your message against the rules
3. If valid ✅ → commit proceeds
4. If invalid ❌ → commit is rejected with error details

## 💡 Tips for Good Commits

### DO:
- Use imperative mood ("add" not "added" or "adds")
- Start with lowercase type
- Be specific and descriptive
- Reference issue numbers when applicable
- Keep the header under 100 characters

### DON'T:
- Use generic messages like "fix stuff" or "updates"
- Include file names (git already tracks this)
- Use past tense ("fixed" → "fix")
- End the subject with a period
- Exceed character limits

## 🛠️ Configuration Files

The commit linting is configured in:
- **`.commitlintrc.js`** - Defines the rules and formats
- **`.husky/commit-msg`** - Git hook that runs the validation
- **`package.json`** - Contains the required dependencies

## 📚 Learn More

- [Conventional Commits Specification](https://www.conventionalcommits.org/)
- [Commitlint Documentation](https://commitlint.js.org/)
- [Husky Documentation](https://typicode.github.io/husky/)

---

**Need help?** If your commit is rejected, read the error message carefully - it will tell you exactly what's wrong and how to fix it!