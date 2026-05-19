"""
UPS Subscription storage using Orthanc Key-Value Store
Implements RAD-86 subscription registry
"""

import orthanc
import json


class UPSSubscriptionStorage:
    """
    Store/retrieve UPS subscriptions using Orthanc Key-Value Store
    Manages which subscribers should be notified for which workitems
    """

    BUCKET = "ups_subscriptions"
    KEY_PREFIX = "subscription:"  # Format: subscription:{workitem_uid}:{subscriber_url}
    GLOBAL_KEY = "global_subscriptions"  # List of global subscribers

    def add_subscription(self, workitem_uid, subscriber_url, deletion_lock=False):
        """
        Add a subscription for a specific workitem

        Args:
            workitem_uid: The workitem UID
            subscriber_url: Subscriber's callback URL
            deletion_lock: Whether subscriber requests deletion lock
        """
        key = f"{self.KEY_PREFIX}{workitem_uid}:{subscriber_url}"
        subscription_data = {
            "workitem_uid": workitem_uid,
            "subscriber_url": subscriber_url,
            "deletion_lock": deletion_lock
        }
        orthanc.StoreKeyValue(self.BUCKET, key, json.dumps(subscription_data).encode('utf-8'))
        print(f"Added subscription: {subscriber_url} -> workitem {workitem_uid}")

    def remove_subscription(self, workitem_uid, subscriber_url):
        """Remove a subscription"""
        key = f"{self.KEY_PREFIX}{workitem_uid}:{subscriber_url}"
        try:
            orthanc.DeleteKeyValue(self.BUCKET, key)
            print(f"Removed subscription: {subscriber_url} from workitem {workitem_uid}")
        except Exception as e:
            print(f"Error removing subscription: {str(e)}")

    def get_subscribers(self, workitem_uid):
        """
        Get all subscriber URLs for a workitem

        Returns:
            List of subscriber URL strings
        """
        subscribers = []

        # Use iterator to find all subscriptions for this workitem
        try:
            it = orthanc.CreateKeysValuesIterator(self.BUCKET)
            while it.Next():
                key = it.GetKey()
                if key.startswith(f"{self.KEY_PREFIX}{workitem_uid}:"):
                    value = it.GetValue()
                    if value:
                        data = json.loads(value.decode('utf-8'))
                        subscribers.append(data['subscriber_url'])
        except Exception as e:
            print(f"Error getting subscribers for {workitem_uid}: {str(e)}")

        # Add global subscribers
        try:
            global_value = orthanc.GetKeyValue(self.BUCKET, self.GLOBAL_KEY)
            if global_value:
                global_subs = json.loads(global_value.decode('utf-8'))
                subscribers.extend(global_subs)
        except:
            pass  # No global subscriptions

        # Remove duplicates
        return list(set(subscribers))

    def add_global_subscription(self, subscriber_url):
        """Add a global subscription (notified for all workitems)"""
        try:
            value = orthanc.GetKeyValue(self.BUCKET, self.GLOBAL_KEY)
            global_subs = json.loads(value.decode('utf-8')) if value else []
        except:
            global_subs = []

        if subscriber_url not in global_subs:
            global_subs.append(subscriber_url)
            orthanc.StoreKeyValue(self.BUCKET, self.GLOBAL_KEY, json.dumps(global_subs).encode('utf-8'))
            print(f"Added global subscription: {subscriber_url}")


# Global instance
subscription_storage = UPSSubscriptionStorage()
