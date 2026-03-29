from __future__ import annotations

import zipfile
from pathlib import Path
from time import monotonic

import boto3
from SoccerNet.Downloader import SoccerNetDownloader

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DATA_ROOT = Path("data/SoccerNet")
TASK_NAME = "SpiideoSynLoc"
SPLITS = ["train", "valid", "test", "challenge"]
VERSION = "4K"


def is_valid_zip(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            return zf.testzip() is None
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile):
        return False


class SimpleProgressBar:
    def __init__(self, total: int | None, desc: str) -> None:
        self.total = total
        self.desc = desc
        self.current = 0
        self.last_percent = -1
        self.started_at = monotonic()

    @staticmethod
    def _format_eta(seconds: float) -> str:
        seconds = max(0, int(seconds))
        hours, rem = divmod(seconds, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def update(self, size: int) -> None:
        self.current += size
        elapsed = max(monotonic() - self.started_at, 1e-6)
        if not self.total:
            speed = self.current / elapsed / (1024 ** 2)
            print(
                f"\r{self.desc}: {self.current / (1024 ** 2):.1f} MiB "
                f"at {speed:.1f} MiB/s",
                end="",
                flush=True,
            )
            return
        percent = int(self.current * 100 / self.total)
        if percent != self.last_percent:
            self.last_percent = percent
            remaining = max(self.total - self.current, 0)
            eta_seconds = remaining / max(self.current / elapsed, 1e-6)
            print(
                f"\r{self.desc}: {percent:3d}% "
                f"({self.current / (1024 ** 3):.2f}/{self.total / (1024 ** 3):.2f} GiB) "
                f"eta {self._format_eta(eta_seconds)}",
                end="",
                flush=True,
            )

    def close(self) -> None:
        print()


def make_progress_bar(total: int | None, desc: str):
    if tqdm is not None:
        return tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=desc)
    return SimpleProgressBar(total=total, desc=desc)


class SafeSoccerNetDownloader(SoccerNetDownloader):
    """Avoid treating interrupted downloads as completed archives."""

    def spiideoDownload(self, path_local, key, verbose=True):  # noqa: N802
        target = Path(path_local)
        partial = target.with_suffix(target.suffix + ".part")

        if is_valid_zip(target):
            if verbose:
                print(f"{target} already exists and passed zip validation")
            return 2

        if target.exists():
            if verbose:
                print(f"{target} exists but is invalid; removing and retrying")
            target.unlink()

        if partial.exists():
            if verbose:
                print(f"{partial} exists from a previous interrupted run; removing it")
            partial.unlink()

        target.parent.mkdir(parents=True, exist_ok=True)

        region = "eu-west-1"
        user_pool_login_id = "cognito-idp.eu-west-1.amazonaws.com/eu-west-1_OKL182JmE"
        client_id = "3v39pho25o0djanccs7b8hdccb"
        identity_pool = "eu-west-1:ef3a2391-2f9c-48bb-8eb0-2e2ec31d259f"
        bucket = "research-data.eu-west-1.prod.spiideo"

        user, password = self.getSpiideoCredentials()
        auth_data = {"USERNAME": user, "PASSWORD": password}
        provider_client = boto3.client("cognito-idp", region_name=region)
        resp = provider_client.initiate_auth(
            AuthFlow="USER_PASSWORD_AUTH",
            AuthParameters=auth_data,
            ClientId=client_id,
        )
        jwt = resp["AuthenticationResult"]["IdToken"]

        client = boto3.client("cognito-identity", region)
        response = client.get_id(
            IdentityPoolId=identity_pool,
            Logins={user_pool_login_id: jwt},
        )

        resp = client.get_credentials_for_identity(
            IdentityId=response["IdentityId"],
            Logins={user_pool_login_id: jwt},
        )

        s3 = boto3.resource(
            "s3",
            aws_access_key_id=resp["Credentials"]["AccessKeyId"],
            aws_secret_access_key=resp["Credentials"]["SecretKey"],
            aws_session_token=resp["Credentials"]["SessionToken"],
            region_name=region,
        )
        obj = s3.Object(bucket_name=bucket, key=key)
        response = obj.get()
        total_size = response.get("ContentLength")
        progress = make_progress_bar(total=total_size, desc=target.name)

        try:
            with open(partial, "wb") as fd:
                for buf in response["Body"].iter_chunks():
                    fd.write(buf)
                    progress.update(len(buf))
            if not is_valid_zip(partial):
                raise zipfile.BadZipFile(f"incomplete or invalid zip downloaded: {partial}")
            partial.replace(target)
        except BaseException:
            if partial.exists():
                partial.unlink()
            raise
        finally:
            progress.close()

        return 0


def main() -> None:
    downloader = SafeSoccerNetDownloader(LocalDirectory=str(DATA_ROOT))
    downloader.downloadDataTask(task=TASK_NAME, split=SPLITS, version=VERSION)


if __name__ == "__main__":
    main()
