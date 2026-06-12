"""
UPS Workitem storage using Orthanc Key-Value Store
Reference: https://orthanc.uclouvain.be/book/plugins/python.html#using-key-value-stores-and-queues-new-in-6-0
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import orthanc

if TYPE_CHECKING:
    from ups.workitem import UPSWorkitem


class UPSStorage:
    """
    Store/retrieve UPS workitems using Orthanc Key-Value Store
    """

    BUCKET = "ups"  # Bucket name for key-value store
    KEY_PREFIX = "upsworkitem"
    INDEX_KEY = "upsworkitemindex"  # List of all workitem UIDs

    def store_workitem(self, workitem: UPSWorkitem) -> None:
        """
        Store workitem in K-V store

        Args:
            workitem: UPSWorkitem instance
        """
        key = f"{self.KEY_PREFIX}{workitem.workitem_uid}"

        # Store workitem data (must be bytes)
        orthanc.StoreKeyValue(self.BUCKET, key, workitem.to_json().encode("utf-8"))

        # Update index
        self._add_to_index(workitem.workitem_uid)

        print(f"Stored workitem {workitem.workitem_uid} with state {workitem.get_state()}")

    def get_workitem(self, workitem_uid: str) -> UPSWorkitem | None:
        """
        Retrieve workitem from K-V store

        Args:
            workitem_uid: The workitem UID

        Returns:
            UPSWorkitem instance or None if not found
        """
        key = f"{self.KEY_PREFIX}{workitem_uid}"

        try:
            value = orthanc.GetKeyValue(self.BUCKET, key)
            if value is None:
                return None

            json_str = value.decode("utf-8")
            from ups.workitem import UPSWorkitem

            return UPSWorkitem.from_json(json_str, workitem_uid)
        except Exception as e:
            print(f"Error retrieving workitem {workitem_uid}: {e!s}")
            return None

    def delete_workitem(self, workitem_uid: str) -> None:
        """
        Delete workitem from K-V store

        Args:
            workitem_uid: The workitem UID
        """
        key = f"{self.KEY_PREFIX}{workitem_uid}"
        try:
            orthanc.DeleteKeyValue(self.BUCKET, key)
            self._remove_from_index(workitem_uid)
            print(f"Deleted workitem {workitem_uid}")
        except Exception as e:
            print(f"Error deleting workitem {workitem_uid}: {e!s}")

    def list_workitems(self, state: str | None = None) -> list[UPSWorkitem]:
        """
        List all workitems, optionally filtered by state

        Args:
            state: Optional state filter (SCHEDULED, IN_PROGRESS, COMPLETED, CANCELED)

        Returns:
            List of UPSWorkitem instances
        """
        workitem_uids = self._get_index()
        workitems: list[UPSWorkitem] = []

        for uid in workitem_uids:
            workitem = self.get_workitem(uid)
            if workitem and (state is None or workitem.get_state() == state):
                workitems.append(workitem)

        return workitems

    def _get_index(self) -> list[str]:
        """Get list of all workitem UIDs"""
        try:
            value = orthanc.GetKeyValue(self.BUCKET, self.INDEX_KEY)
            if value is None:
                return []
            return cast("list[str]", json.loads(value.decode("utf-8")))
        except Exception:
            return []

    def _add_to_index(self, workitem_uid: str) -> None:
        """Add workitem UID to index"""
        index = self._get_index()
        if workitem_uid not in index:
            index.append(workitem_uid)
            orthanc.StoreKeyValue(self.BUCKET, self.INDEX_KEY, json.dumps(index).encode("utf-8"))

    def _remove_from_index(self, workitem_uid: str) -> None:
        """Remove workitem UID from index"""
        index = self._get_index()
        if workitem_uid in index:
            index.remove(workitem_uid)
            orthanc.StoreKeyValue(self.BUCKET, self.INDEX_KEY, json.dumps(index).encode("utf-8"))


# Global instance
ups_storage = UPSStorage()
