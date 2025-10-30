# PDP-Analysis – Scripts

This repository contains the **base PDP code** as well as the **individual versions** of the code developed by different researchers. The goal is to provide a clean and flexible way to collaborate while allowing everyone to experiment with their own code.

---

## Repository Structure
scripts/  
│  
├── base code/ # The official base PDP code  
├── Bart's code/ # Bart's personal version of the PDP code  
├── CB's code/ # CB's personal version of the PDP code  
├── Jana's code/ # Jana's personal version of the PDP code  
├── Olivier's code/ # Olivier's personal version of the PDP code  

---

## Workflow

1. **Start from the base code**  
   - The `base code` folder contains the most up-to-date (common) version of the PDP code.
   - Every researcher’s folder (`Bart's code`, `CB's code`, etc.) contains their own working copy, which may include experiments, modifications, or extensions.

2. **Develop your own version**  
   - Fork this repository.  
   - Work in your **own folder** (e.g., `Jana's code/`) without touching the `base code` directly.

3. **Contribute improvements to the base code**  
   - If you find a useful update, fix, or extension:
     - Add or update the relevant scripts in the `base code` folder.
     - Push these changes back to the repository.  
   - This way, the `base code` always evolves with the best contributions from all researchers.

4. **Synchronize your own version**  
   - When the `base code` is updated, you can:
     - Download the repository.
     - Use **VSCode Copilot** (or another diff/merge tool) to compare the `base code` with your personal folder.  
       Since both the base and all researcher versions exist in the same repo, VSCode can read them all and help you adjust your code accordingly.

---

## Collaboration Guidelines

- **Do not directly edit another researcher’s folder.** Only work in your own folder and in the `base code` folder when you want to contribute.
- **Keep the base code clean.** Only push tested and stable code to the `base code` folder.
- **Experiment freely in your own folder.** This is your personal sandbox.

---

## Example Workflow

1. Clone or fork the repository.
2. Add your experiments to your personal folder (e.g., `Jana's code/`).
3. Discover an improvement? → Update the scripts in `base code/`.
4. Commit and push.
5. Others can pull the repository, then merge updates from `base code` into their own version with VSCode Copilot’s assistance.

---

## Benefits

- Everyone can **work independently** without conflicts.
- The **base code evolves collaboratively** with the best contributions.
- Synchronization is simple thanks to **Copilot’s diff/merge support**.
- The setup is **transparent**: all versions + base version are in one repository.

---

🚀 Happy coding and analyzing PDP together!
