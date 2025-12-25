import subprocess
import os

"""
References::
https://gitpython.readthedocs.io/en/stable/tutorial.html
"""
def get_git_info(repo_path='.'):
    """
    Retrieves git information: commit hash, remote URL, and branch name.
    
    Args:
        repo_path (str): Path to the git repository. Defaults to current directory.
        
    Returns:
        dict: Dictionary containing 'commit_hash', 'remote_url', and 'branch_name'.
              Values are None if retrieval fails.
    """
    def run_git_command(command):
        try:
            result = subprocess.run(
                command, 
                cwd=repo_path, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True, 
                check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    commit_hash = run_git_command(['git', 'rev-parse', 'HEAD'])
    remote_url = run_git_command(['git', 'config', '--get', 'remote.origin.url'])
    branch_name = run_git_command(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])

    return {
        'commit_hash': commit_hash,
        'remote_url': remote_url,
        'branch_name': branch_name
    }

def print_git_info(repo_path='.'):
    """
    Prints git information to stdout.
    """
    info = get_git_info(repo_path)
    print("-" * 30)
    print("Git Repository Information:")
    if info['commit_hash']:
        print(f"Commit Hash: {info['commit_hash']}")
    else:
        print("Commit Hash: Not available")
        
    if info['branch_name']:
        print(f"Branch Name: {info['branch_name']}")
    else:
        print("Branch Name: Not available")

    if info['remote_url']:
        print(f"Remote URL:  {info['remote_url']}")
    else:
        print("Remote URL:  Not available")
    print("-" * 30)

if __name__ == "__main__":
    print_git_info()
