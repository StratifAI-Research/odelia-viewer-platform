"""
UPS Workitem storage using Orthanc Key-Value Store
Reference: https://orthanc.uclouvain.be/book/plugins/python.html#using-key-value-stores-and-queues-new-in-6-0
"""

import orthanc
import json


class UPSStorage:
    """
    Store/retrieve UPS workitems using Orthanc Key-Value Store
    """

    BUCKET = "ups"  # Bucket name for key-value store
    KEY_PREFIX = "upsworkitem"
    INDEX_KEY = "upsworkitemindex"  # List of all workitem UIDs

    def store_workitem(self, workitem):
        """
        Store workitem in K-V store

        Args:
            workitem: UPSWorkitem instance
        """
        key = f"{self.KEY_PREFIX}{workitem.workitem_uid}"

        # Store workitem data (must be bytes)
        orthanc.StoreKeyValue(self.BUCKET, key, workitem.to_json().encode('utf-8'))

        # Update index
        self._add_to_index(workitem.workitem_uid)

        print(f"Stored workitem {workitem.workitem_uid} with state {workitem.get_state()}")

    def get_workitem(self, workitem_uid):
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

            json_str = value.decode('utf-8')
            from ups.workitem import UPSWorkitem
            return UPSWorkitem.from_json(json_str, workitem_uid)
        except Exception as e:
            print(f"Error retrieving workitem {workitem_uid}: {str(e)}")
            return None

    def delete_workitem(self, workitem_uid):
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
            print(f"Error deleting workitem {workitem_uid}: {str(e)}")

    def list_workitems(self, state=None):
        """
        List all workitems, optionally filtered by state

        Args:
            state: Optional state filter (SCHEDULED, IN_PROGRESS, COMPLETED, CANCELED)

        Returns:
            List of UPSWorkitem instances
        """
        workitem_uids = self._get_index()
        workitems = []

        for uid in workitem_uids:
            workitem = self.get_workitem(uid)
            if workitem:
                if state is None or workitem.get_state() == state:
                    workitems.append(workitem)

        return workitems

    def _get_index(self):
        """Get list of all workitem UIDs"""
        try:
            value = orthanc.GetKeyValue(self.BUCKET, self.INDEX_KEY)
            if value is None:
                return []
            return json.loads(value.decode('utf-8'))
        except:
            return []

    def _add_to_index(self, workitem_uid):
        """Add workitem UID to index"""
        index = self._get_index()
        if workitem_uid not in index:
            index.append(workitem_uid)
            orthanc.StoreKeyValue(self.BUCKET, self.INDEX_KEY, json.dumps(index).encode('utf-8'))

    def _remove_from_index(self, workitem_uid):
        """Remove workitem UID from index"""
        index = self._get_index()
        if workitem_uid in index:
            index.remove(workitem_uid)
            orthanc.StoreKeyValue(self.BUCKET, self.INDEX_KEY, json.dumps(index).encode('utf-8'))


# Global instance
ups_storage = UPSStorage()
