import os
import sys
import time
import logging
import subprocess
import json
from datetime import datetime
from pathlib import Path
import git
import docker

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
REPO_PATH = Path(os.getenv("REPO_PATH", "/repo"))
POLL_INTERVAL = int(os.getenv("GITOPS_POLL_INTERVAL", "30"))
ENABLE_AUTO_PULL = os.getenv("GITOPS_ENABLE_AUTO_PULL", "true").lower() == "true"
ENABLE_AUTO_DEPLOY = os.getenv("GITOPS_ENABLE_AUTO_DEPLOY", "true").lower() == "true"
ENABLE_AUTO_COMMIT = os.getenv("GITOPS_ENABLE_AUTO_COMMIT", "true").lower() == "true"
ENABLE_AUTO_PUSH = os.getenv("GITOPS_ENABLE_AUTO_PUSH", "false").lower() == "true"
GIT_USER_NAME = os.getenv("GIT_USER_NAME", "AutoGrandmaster Bot")
GIT_USER_EMAIL = os.getenv("GIT_USER_EMAIL", "bot@autograndmaster.local")

# Files that should never be committed
IGNORED_FILES = ['.env', '*.env.local', '*.log', 'models/*.pt', 'data/games/*', 'data/metrics/*']

# Initialize Docker client
try:
    docker_client = docker.from_env()
    logger.info("Docker client initialized")
except Exception as e:
    logger.error(f"Failed to initialize Docker client: {e}")
    docker_client = None

class GitOpsManager:
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.repo = None
        self.last_commit = None
        
        try:
            self.repo = git.Repo(repo_path)
            logger.info(f"Git repository initialized at: {repo_path}")
            
            # Configure git user
            with self.repo.config_writer() as config:
                config.set_value("user", "name", GIT_USER_NAME)
                config.set_value("user", "email", GIT_USER_EMAIL)
            
            self.last_commit = self.repo.head.commit.hexsha
            logger.info(f"Current commit: {self.last_commit[:8]}")
        except Exception as e:
            logger.error(f"Failed to initialize git repository: {e}")
    
    def check_for_updates(self) -> bool:
        """Check if there are new commits in remote"""
        if not self.repo:
            return False
        
        try:
            # Fetch from remote
            origin = self.repo.remotes.origin
            origin.fetch()
            
            # Get remote commit
            remote_commit = origin.refs[self.repo.active_branch.name].commit.hexsha
            
            if remote_commit != self.last_commit:
                logger.info(f"New commits detected: {self.last_commit[:8]} -> {remote_commit[:8]}")
                return True
            
            return False
        except Exception as e:
            logger.error(f"Failed to check for updates: {e}")
            return False
    
    def pull_changes(self) -> bool:
        """Pull latest changes from remote"""
        if not self.repo:
            return False
        
        try:
            logger.info("Pulling latest changes...")
            origin = self.repo.remotes.origin
            pull_info = origin.pull()
            
            for info in pull_info:
                logger.info(f"Pulled: {info.ref} -> {info.commit}")
            
            self.last_commit = self.repo.head.commit.hexsha
            logger.info(f"Pull completed. New commit: {self.last_commit[:8]}")
            return True
        except Exception as e:
            logger.error(f"Failed to pull changes: {e}")
            return False
    
    def has_local_changes(self) -> bool:
        """Check if there are local uncommitted changes"""
        if not self.repo:
            return False
        
        try:
            # Check for untracked files and modified files
            return self.repo.is_dirty() or len(self.repo.untracked_files) > 0
        except Exception as e:
            logger.error(f"Failed to check local changes: {e}")
            return False
    
    def commit_and_push_changes(self, message: str = None) -> bool:
        """Commit and push local changes"""
        if not self.repo:
            return False
        
        try:
            # Check if there are changes
            if not self.has_local_changes():
                logger.info("No local changes to commit")
                return False
            
            # Get list of changed files (excluding ignored patterns)
            changed_files = []
            
            # Add modified tracked files
            for item in self.repo.index.diff(None):
                if not any(self._matches_pattern(item.a_path, pattern) for pattern in IGNORED_FILES):
                    changed_files.append(item.a_path)
            
            # Add untracked files
            for file in self.repo.untracked_files:
                if not any(self._matches_pattern(file, pattern) for pattern in IGNORED_FILES):
                    changed_files.append(file)
            
            if not changed_files:
                logger.info("No changes to commit (all ignored)")
                return False
            
            logger.info(f"Staging {len(changed_files)} files...")
            for file in changed_files:
                self.repo.index.add([file])
            
            # Generate commit message if not provided
            if not message:
                message = self._generate_commit_message(changed_files)
            
            # Commit
            commit = self.repo.index.commit(message)
            logger.info(f"Committed: {commit.hexsha[:8]} - {message}")
            
            # Push if enabled
            if ENABLE_AUTO_PUSH:
                logger.info("Pushing to remote...")
                origin = self.repo.remotes.origin
                origin.push()
                logger.info("Push completed")
            else:
                logger.info("Auto-push disabled, skipping push")
            
            return True
        except Exception as e:
            logger.error(f"Failed to commit/push changes: {e}")
            return False
    
    def _matches_pattern(self, filepath: str, pattern: str) -> bool:
        """Check if filepath matches ignore pattern"""
        if '*' in pattern:
            # Simple wildcard matching
            if pattern.startswith('*.'):
                return filepath.endswith(pattern[1:])
            elif pattern.endswith('/*'):
                return filepath.startswith(pattern[:-2])
            else:
                return pattern.replace('*', '') in filepath
        return filepath == pattern
    
    def _generate_commit_message(self, changed_files: list) -> str:
        """Generate conventional commit message"""
        # Categorize changes
        docs = [f for f in changed_files if f.endswith('.md') or 'docs/' in f]
        config = [f for f in changed_files if f.endswith('.yml') or f.endswith('.yaml') or f.endswith('.json')]
        code = [f for f in changed_files if f.endswith(('.py', '.js', '.jsx', '.ts', '.tsx'))]
        
        if docs and not code and not config:
            return f"docs: Update documentation ({len(docs)} files)"
        elif config and not code:
            return f"chore: Update configuration ({len(config)} files)"
        elif code:
            return f"feat: Update codebase ({len(code)} files)"
        else:
            return f"chore: Update {len(changed_files)} files"
    
    def update_changelog(self):
        """Update CHANGELOG.md with latest changes"""
        try:
            changelog_path = self.repo_path / "CHANGELOG.md"
            
            # Read existing changelog
            if changelog_path.exists():
                with open(changelog_path, 'r') as f:
                    content = f.read()
            else:
                content = "# Changelog\n\n"
            
            # Generate new entry
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            entry = f"## {timestamp}\n\n"
            
            # Get recent commits
            commits = list(self.repo.iter_commits(max_count=5))
            for commit in commits:
                entry += f"- {commit.message.strip()} ({commit.hexsha[:8]})\n"
            
            entry += "\n"
            
            # Insert new entry after header
            lines = content.split('\n')
            header_end = 2  # After "# Changelog\n\n"
            new_content = '\n'.join(lines[:header_end]) + '\n' + entry + '\n'.join(lines[header_end:])
            
            # Write back
            with open(changelog_path, 'w') as f:
                f.write(new_content)
            
            logger.info("Changelog updated")
        except Exception as e:
            logger.error(f"Failed to update changelog: {e}")

class DeploymentManager:
    def __init__(self):
        self.client = docker_client
    
    def rebuild_and_deploy(self) -> bool:
        """Rebuild and redeploy services"""
        if not self.client:
            logger.error("Docker client not available")
            return False
        
        try:
            logger.info("Starting rebuild and redeploy...")
            
            # Run docker compose commands
            result = subprocess.run(
                ["docker", "compose", "build"],
                cwd="/repo",
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Build failed: {result.stderr}")
                return False
            
            logger.info("Build completed successfully")
            
            # Redeploy
            result = subprocess.run(
                ["docker", "compose", "up", "-d", "--remove-orphans"],
                cwd="/repo",
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Deploy failed: {result.stderr}")
                return False
            
            logger.info("Deploy completed successfully")
            
            # Wait for health checks
            time.sleep(10)
            
            if not self.check_services_health():
                logger.warning("Some services may not be healthy")
            
            return True
        except Exception as e:
            logger.error(f"Failed to rebuild/deploy: {e}")
            return False
    
    def check_services_health(self) -> bool:
        """Check if services are healthy"""
        if not self.client:
            return False
        
        try:
            containers = self.client.containers.list(
                filters={"label": "com.docker.compose.project=autograndmaster"}
            )
            
            all_healthy = True
            for container in containers:
                status = container.status
                logger.info(f"Container {container.name}: {status}")
                if status != "running":
                    all_healthy = False
            
            return all_healthy
        except Exception as e:
            logger.error(f"Failed to check service health: {e}")
            return False

def main():
    logger.info("=== AutoGrandmaster GitOps Service Starting ===")
    logger.info(f"Repository: {REPO_PATH}")
    logger.info(f"Poll interval: {POLL_INTERVAL}s")
    logger.info(f"Auto-pull: {ENABLE_AUTO_PULL}")
    logger.info(f"Auto-deploy: {ENABLE_AUTO_DEPLOY}")
    logger.info(f"Auto-commit: {ENABLE_AUTO_COMMIT}")
    logger.info(f"Auto-push: {ENABLE_AUTO_PUSH}")
    
    gitops = GitOpsManager(REPO_PATH)
    deploy = DeploymentManager()
    
    logger.info("GitOps service initialized, entering main loop...")
    
    while True:
        try:
            # Check for remote updates
            if ENABLE_AUTO_PULL and gitops.check_for_updates():
                logger.info("Remote updates detected")
                
                if gitops.pull_changes():
                    logger.info("Changes pulled successfully")
                    
                    # Update changelog
                    gitops.update_changelog()
                    
                    # Auto-deploy if enabled
                    if ENABLE_AUTO_DEPLOY:
                        logger.info("Starting auto-deploy...")
                        if deploy.rebuild_and_deploy():
                            logger.info("Auto-deploy successful")
                        else:
                            logger.error("Auto-deploy failed")
            
            # Check for local changes
            if ENABLE_AUTO_COMMIT and gitops.has_local_changes():
                logger.info("Local changes detected")
                gitops.commit_and_push_changes()
            
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        
        # Wait before next check
        time.sleep(POLL_INTERVAL)
    
    logger.info("GitOps service stopped")

if __name__ == "__main__":
    main()
